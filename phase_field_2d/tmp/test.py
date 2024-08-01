from typing import NewType, Any
import numpy as np
from numpy.typing import NDArray

x: NDArray[Any]


UserId = NewType('UserId', int)
some_id = UserId(524313)
#%%

from functools import partial
from pipetools import pipe

odd_sum = pipe | range | partial(filter, lambda x: x % 2) | sum

odd_sum(10)  # -> 25
odd_sum = pipe | range | where(lambda x: x % 2) | sum