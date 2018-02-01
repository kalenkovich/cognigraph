def class_name_of(obj: object):
    return type(obj).__name__


def all_upper(iterable_of_strings):
    """
    Converts all the strings to upper case
    :param iterable_of_strings:
    :return: iterable of the same type
    """
    iterable_type = type(iterable_of_strings)
    list_all_upper = [s.upper() for s in iterable_of_strings]
    return iterable_type(list_all_upper)
