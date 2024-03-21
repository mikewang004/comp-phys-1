# comp-phys-1
Simulation for the three matters of state of Argon. The relevant input parameters are located at the bottom of 
main4.py as follows:

h: step size, recommended to be <0.01
max_time: simulation time.
method: Can be "verlet" or "euler", verlet recommended for proper simulation. 
density: in unitless.
temperature: in unitless.

Generation is done by switching on 
    sim1.system.generate_fcc_lattice()
    sim1.system.generate_velocities()
    sim1.run_simulation(h=h, max_time=max_time, method=method)

and results can be obtained by plotting 
    sim1.animate_sim_results(frame_skip_multiplier=1)
    sim1.plot_system_energy()

for respectively an animation of the particle evolution and the total energy in the system. 

