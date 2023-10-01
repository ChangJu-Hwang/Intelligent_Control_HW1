import os

from matplotlib import pyplot as plt

import yaml

script_dir = os.path.join(os.path.dirname(__file__))
log_dir = os.path.join(os.path.dirname(script_dir), 'log', 'log.yaml')

with open(log_dir) as log:
    log = yaml.load(log, Loader=yaml.Loader)
    legend = []
    for key, value in sorted(log.items()):
        x = range(len(value))
        y = value
        legend.append(str(key) + "th Data")
        plt.plot(x, y)

plt.title('Sum of squared errors for each output unit',
          fontweight="bold")
plt.xlabel('Epochs',
           fontweight="bold")
plt.ylabel('Error',
           fontweight="bold")
plt.legend(legend)

plt.xlim(left = 0.0)
# plt.ylim(bottom = 0.0)
plt.ylim(bottom = 0.0, top = 1.0)

plt.show()