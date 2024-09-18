Examples
========

.. code-block:: python

    # Example 1: Simple addition
    def add(a, b):
        return a + b

    print(add(2, 3))  # Output: 5

.. code-block:: python

    # Example 2: Class definition
    class Dog:
        def __init__(self, name):
            self.name = name

        def bark(self):
            return f"{self.name} says woof!"

    my_dog = Dog("Buddy")
    print(my_dog.bark())  # Output: Buddy says woof!