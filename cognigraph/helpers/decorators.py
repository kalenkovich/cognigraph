# accepts and returns are adapted from https://www.python.org/dev/peps/pep-0318/


def accepts(*types):
    def check_accepts(f):
        assert len(types) == f.__code__.co_argcount

        def new_f(*args, **kwargs):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), \
                       "arg %r does not match %s" % (a, t)
            return f(*args, **kwargs)
        new_f.__name__ = f.__name__
        return new_f
    return check_accepts


def returns(rtype):
    def check_returns(f):
        def new_f(*args, **kwargs):
            result = f(*args, **kwargs)
            assert isinstance(result, rtype), \
                "return value %r does not match %s" % (result, rtype)
            return result
        new_f.__name__ = f.__name__
        return new_f
    return check_returns
