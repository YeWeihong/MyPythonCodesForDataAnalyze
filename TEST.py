from spectrum_toolbox import MDSDataLoader
import matplotlib.pyplot as plt
import numpy as np
loader = MDSDataLoader()
time, wmhd = loader.get_signal(158900, '\\wmhd', time_range=(2.8, 3.4), tree='efitrt_east')
fig = plt.figure(figsize=(10, 4))
plt.plot(time, wmhd/1000, color='red', linewidth=2)
plt.show()