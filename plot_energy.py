from main3 import *

# Note for loop_results_e 2nd axis 0th index contains kinetic energy whereas
# 1st index contains potential energy



time_array = np.arange(0, int(max_time/h))
# Try energy plot for one particle 


plt.scatter(time_array, loop_results_e[:, 1, 0], label = "kinetic", marker=".")
plt.scatter(time_array, loop_results_e[:, 1, 1], label = "potential", marker = ".")
#plt.xlim(10, int(max_time/h))
#plt.scatter(time_array, np.sum(loop_results_e[:, 0, :], axis = 1), label = "total", marker=".")
plt.legend()
plt.xlabel("time")
plt.ylabel("energy")
plt.title("Energy of one argon atom over time")
plt.show()

