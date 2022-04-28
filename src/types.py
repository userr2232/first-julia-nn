from enum import Enum


class Month(Enum):
    MARCH = 3
    JUNE = 6
    SEPTEMBER = 9
    DECEMBER = 12

    def describe(self):
        return (self.name + " " + ("equinox" if self.value in [3, 9] else "solstice")).title()

    def __str__(self):
        return f"{self.value:02d}"