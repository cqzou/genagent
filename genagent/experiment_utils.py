from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from itertools import product
from frozendict import frozendict

def run_parallel_dict_map(
    func,
    items,
    max_workers: int = 5,
    key_map=None,
):
    """
    Like map, but uses a ThreadPoolExecutor.
    """
    if key_map is None:
        key_map = lambda x: x  # noop
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = dict(
            tqdm(
                executor.map(
                    lambda item: (
                        key_map(item),
                        func(item),
                    ),
                    items,
                ),
                total=len(items),
            )
        )
    return results

def run_parallel_dict_product(
    func,
    **kwargs,
):
    keys = kwargs.keys()
    args = [
        frozendict(zip(keys, p, strict=True)) for p in product(*kwargs.values())
    ]
    def _inner(args):
        return func(**args)
    return run_parallel_dict_map(_inner, args)