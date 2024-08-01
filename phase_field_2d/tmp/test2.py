#%%
from phase_field_2d import PhaseField
from phase_field_2d import get_matrix_image

test = PhaseField(4, 700, 0.5)
test.dtime = 0.01
test.start()

# %%
