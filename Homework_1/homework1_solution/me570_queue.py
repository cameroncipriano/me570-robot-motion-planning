"""
A pedagogical implementation of a priority queue
"""


class PriorityElement:
    """ Store a key and a value about an element of the queue. """
    def __init__(self, key, value):
        """
        Stores the arguments as internal attributes.
        """
        self.element = (key, value)

    def __str__(self) -> str:
        return f"{self.element}"


class Priority:
    """ Implements a priority queue """
    def __init__(self):
        """
        Initializes the internal attribute  queue to be an empty list.
        """
        self.queue = []

    def insert(self, key, cost):
        """
        Add an element to the queue.
        """
        self.queue.append(PriorityElement(key, cost))

    def min_extract(self):
        """
        Extract the element with minimum cost from the queue.
        """
        if len(self.queue) == 0:
            return None, None

        min_location = 0
        for i, p_element in enumerate(self.queue):
            if p_element.element[1] < self.queue[min_location].element[1]:
                min_location = i

        key = self.queue[min_location].element[0]
        cost = self.queue[min_location].element[1]

        del self.queue[min_location]

        return key, cost

    def is_member(self, key):
        """
        Check whether an element with a given key is in the queue or not.
        """
        flag = False

        for _, p_element in enumerate(self.queue):
            if p_element.element[0] == key:
                flag = True
                break

        return flag
