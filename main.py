#!/usr/bin/python3

import argparse
from collections import OrderedDict
from enum import Enum
from itertools import takewhile
import random
import re
import string
import sys

from gooey import Gooey, GooeyParser
from jellyfish import jaro_winkler
from ortools.graph import pywrapgraph

re_email = re.compile(r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)')

# The Jaro-Winkler score above which two strings will be considered
# equivalent.
jaro_winkler_threshold = 0.95

def ranks_to_weights(num_topics, ranks):
    '''Convert a list of topics in ranked order to a dictionary of weights'''
    weights = { topic: num_topics - rank for rank, topic in enumerate(ranks) }
    return weights


class State(Enum):
    '''Parser states'''
    TOPICS = 1
    USERNAME = 2
    PREFS = 3


class Prefs:
    '''The list of topics and preferences for those topics.

    Attributes:
        topics   A list of strings containing the name of each topic.
        users    A map of user names to a list of indices into the topics list in order of preference.
    '''
    topics = []
    users = {}

    def __init__(self, topics = [], users = {}):
        self.users = users
        self.topics = topics

    def new_random(topics, students):
        assert students < 26
        assert topics >= students

        users = {}
        for i in range(students):
            name = string.ascii_uppercase[i]
            prefs = random.sample(range(0, topics), 3)
            users[name] = prefs

        topics = [str(i + 1) for i in range(topics)]
        return Prefs(topics, users)

    def from_text(lines):
        self = Prefs()
        state = State.TOPICS
        username = ''
        prefs = [] # The list of prefs for the current user

        # Skip lines at the beginning containing whitespace
        lines = enumerate(lines)
        takewhile(lambda _, s: s.isspace(), lines)

        for i, line in lines:
            if line.isspace():
                # Use a blank line to mark the end of topics and
                # preferences lists
                if state is State.TOPICS:
                    state = State.USERNAME
                elif state is State.PREFS:
                    self.users[username] = prefs
                    prefs = []
                    username = ''
                    state = State.USERNAME

                continue # Ignore lines containing only whitespace

            if state is State.TOPICS:
                self.topics.append(line.strip())

            elif state is State.USERNAME:
                state = State.PREFS
                username = line.strip()

                if re_email.search(username) is None:
                    print('WARNING(Line {}): Username "{}" does not look like an email'.format(i+1, username),
                          file = sys.stderr)

                if username in self.users:
                    err = 'ERROR(Line {}): Username "{}" appears multiple times in the preference list'.format(i+1, username)
                    raise ValueError(err)

            elif state is State.PREFS:
                pref = line.strip()
                topic = self.index_topic(pref)
                if topic is None:
                    err = 'ERROR(Line {}): User preference "{}" does not appear in the topics list'.format(i+1, pref)
                    raise ValueError(err)

                prefs.append(topic)

        if username != '':
            self.users[username] = prefs

        return self

    def from_csv():
        header = f.readline()
        assert header is not None, "Empty csv file"
        f.seek(0)

        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(header)
        has_header = sniffer.has_header(header)
        rows = csv.reader(f, dialect = dialect)

        if has_header:
            next(rows)

        users = {}
        topics = OrderedDict()
        for row in rows:
            # Skip rows with no value in the first column
            username = row[0]
            if username == '':
                continue

            assert username not in users, "Duplicate usernames"

            prefs = []
            for rank, topic in enumerate(row[1:]):
                topic = topics.setdefault(topic, len(topics))
                assert pid not in prefs, "Multiple preferences for the same presentation id"
                prefs.append(topic)

            users[username] = prefs

        return Prefs([topic for topic in topics.keys()], users)

    def index_topic(self, s):
        first, second = None, None
        scores = ((i, jaro_winkler(s, topic)) for i, topic in enumerate(self.topics))
        for i, score in scores:
            # Return early on exact match
            if score == 1.0:
                return i

            if first is None or score > first[1]:
                first, second = (i, score), first
            elif second is None or score > second[1]:
                second = (i, score)

        if first is None:
            return None

        if second is None:
            i, score = first
            if score >= jaro_winkler_threshold:
                return i
            else:
                return None

        if first[1] - second[1] < 0.1 or first[1] < jaro_winkler_threshold:
            return None

        i = first[0]
        print('WARN: Corrected user preference "{}" to topic "{}"'.format(s, self.topics[i]))
        return i

    def print(self, sol):
        print()
        print('Topics:')
        for topic in self.topics:
            print('\t{}'.format(topic))

        print()
        for name, ranks in self.users.items():
            print()
            print(name)
            for i, topic in enumerate(ranks):
                print('\t{}. {}'.format(i+1, self.topics[topic]))

        print()
        for name, topic in sol.items():
            ranks = self.users[name]
            rank = ranks.index(topic) + 1 if topic in ranks else None
            topic_name = self.topics[topic]
            print('"{}" assigned to "{}" (rank = {})'.format(name, topic_name, rank))

    def solve(self):
        size = len(self.topics) # The size of the cost matrix
        students = [(name, ranks_to_weights(size, prefs)) for name, prefs in self.users.items()]

        # Shuffle students so none are given an advantage based on input
        # ordering.
        random.shuffle(students)

        # If we have more presentations than students, add students with
        # no preferences to provide a square cost matrix for the solver
        dummies = max(0, size - len(students))
        for _ in range(dummies):
            students.append((None, {}))

        # At this point the cost matrix must be square
        assert len(students) == size, "More students than topics"

        # Build the graph
        # Students are one set of nodes on the bipartite graph,
        # presentations the other.
        graph = pywrapgraph.LinearSumAssignment()
        for row, (name, weights) in enumerate(students):
            for col in range(size):
                # Invert weight for min-solver
                weight = -weights.get(col, 0)
                graph.AddArcWithCost(row, col, weight)

        status = graph.Solve()
        assert status == graph.OPTIMAL, 'Not all students could be assigned to a presentation'

        sol = { name: graph.RightMate(i) for i, (name, _) in enumerate(students) if name is not None }
        return sol


@Gooey
def main():
    parser = GooeyParser(
            description = 'Assign students to presentation topics')

    # parser.add_argument(
    #         '--test',
    #         action='store_true',
    #         help = 'Runs a test')

    parser.add_argument(
            'input',
            widget = 'FileChooser',
            type = argparse.FileType('r'),
            help = 'A text file containing a list of presentations along with student names and their preferences')

    parser.add_argument(
            '--similarity',
            type = float,
            default = 0.95,
            help = 'A number between 0.7 and 1.0 indicating the tolerance used when comparing strings. 1.0 is strictest.')

    args = parser.parse_args()
    assert 1.0 >= args.similarity >= 0.7, "Similarity must be between 0.7 and 1.0"
    jaro_winkler_threshold = args.similarity
    prefs = Prefs.from_text(args.input)

    sol = prefs.solve()
    prefs.print(sol)

if __name__ == '__main__':
    main()


