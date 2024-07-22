import pdb

def pdb_decorator(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(">>BUG: ", e)
            import pdb; pdb.post_mortem()   # post_mortem() can monitor args which trigger the error, while set_trace() cannot.
    return wrapper

if __name__ == "__main__":
    pass