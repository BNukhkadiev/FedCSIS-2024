from tpot import TPOTClassifier
from dask.distributed import Client
import cost_function
def get_automl():
    client = Client(n_workers=4, threads_per_worker=1)
    return TPOTClassifier(generations=100,population_size=100,verbosity=2,scoring=cost_function.get_scorer(),use_dask=True)