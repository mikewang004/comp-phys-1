from main3 import *

# Note for loop_results_e 2nd axis 0th index contains kinetic energy whereas
# 1st index contains potential energy



time_array = np.arange(0, int(max_time/h))
# Try energy plot for one particle 
print(np.max(np.abs(loop_results_e[:, 0, 1])))

plt.scatter(time_array, loop_results_e[:, 0, 0], label = "kinetic", marker=".")
plt.scatter(time_array, np.abs(loop_results_e[:, 0, 1]), label = "potential", marker = ".")
plt.scatter(time_array, np.sum(np.abs(loop_results_e[:, 0, :]),axis = 1), label = "total", marker = ".")
#plt.scatter(time_array, loop_results_v[:, 1, 0], label = "velocity in x", marker=".")
#plt.scatter(time_array, loop_results_v[:, 1, 1], label = "velocity in y", marker=".")
#plt.scatter(time_array, np.linalg.norm(loop_results_v[:, 0, :], axis = 1), label = "velocity total", marker=".")
plt.legend()
plt.xlabel("time")
plt.ylabel("energy or velocity (refer to label)")
plt.title("Energy of one argon atom over time")
plt.show()



