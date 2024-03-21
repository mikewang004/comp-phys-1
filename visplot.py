import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_results(input_x, input_y, view_size=10, frame_interval=10, trailing_frames=1, frame_skip_multiplier = 1):
    fig, ax = plt.subplots()
    ax.set_xlim([-view_size, view_size])
    ax.set_ylim([-view_size, view_size])

    n_particles = np.shape(input_x)[1]
    n_frames = np.shape(input_x)[0] + 1
    lines = []
    plt.grid()
    # set up first frame for plotting and construct all lines
    for i in range(0, n_particles):
        frame = 0
        plotline = ax.plot(
            input_x[frame, i], input_y[frame, i], marker="o", linestyle="", markersize=2
        )
        lines.append(plotline[0])

    def update(frame):
        frame = frame_skip_multiplier * frame
        for i in range(0, len(lines)):
            line = lines[i]
            trailing_frame = max(0, frame - trailing_frames)
            line.set_xdata(
                input_x[trailing_frame:frame, i],
            )
            line.set_ydata(input_y[trailing_frame:frame, i])
        return lines

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=n_frames,
        interval=frame_interval,
        repeat=True,
        cache_frame_data=False,
    )

    plt.show()


def animate_quiver(
    positions_x,
    positions_y,
    vec_x,
    vec_y,
    view_size=10,
    frame_interval=10,
    arrow_scaling=1,
):
    # n_particles = np.shape(arrow_positions)[1]
    n_frames = np.shape(positions_x)[0]
    print(f"{n_frames=}")

    fig2, ax = plt.subplots(1, 1)
    quiver = ax.quiver(
        positions_x[0, :],  # 0 for first frame
        positions_y[0, :],
        vec_x[0, :],
        vec_y[0, :],
        pivot="tail",
        # color="r",
    )

    ax.set_xlim(-view_size, view_size)
    ax.set_ylim(-view_size, view_size)

    def update_quiver(n_frame, quiver, positions_x, positions_y, vec_x, vec_y):
        """updates the horizontal and vertical vector components by a
        fixed increment on each frame
        """
        print(f"{n_frame=}")
        U = arrow_scaling * vec_x[n_frame, :]
        V = arrow_scaling * vec_y[n_frame, :]

        positions = np.transpose(np.vstack((positions_x[n_frame, :], positions_y[n_frame, :])))
        print(f"{np.shape(positions)=}")
        quiver.set_offsets(positions)
        quiver.set_UVC(U, V)

        return (quiver,)

    anim = animation.FuncAnimation(
        fig2,
        update_quiver,
        fargs=(quiver, positions_x, positions_y, vec_x, vec_y),
        interval=frame_interval,
        blit=False,
        frames=n_frames,
    )
    plt.show()

    return

    

def animate_results3d(input_x, input_y, input_z, view_size=10, frame_interval=10, trailing_frames=1, frame_skip_multiplier = 1):
    n_frames = np.shape(input_x)[0] + 1
    def update_graph(num):
        num = (frame_skip_multiplier * num) % n_frames
        graph._offsets3d = (input_x[num], input_y[num], input_z[num])
        title.set_text('3D Test, time={}'.format(num))
        return


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')

    graph = ax.scatter(input_x, input_y, input_z)

    ani = animation.FuncAnimation(fig, update_graph, n_frames, 
                               interval=40, blit=False)

    plt.show()
    
def make_xyplot(x,y, xlabel = 'time (unitless)', ylabel='', plotname=''):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y, label=plotname)


# animate_results(loop_results_x[:,:,0], loop_results_x[:,:,1], view_size=0.6*L, trailing_frames = 1000)
# animate_quiver(
#     loop_results_x[:, :, 0],
#     loop_results_x[:, :, 1],
#     selected_diff[:,0],
#     selected_diff[:,1],
#     arrow_scaling=1,
#     frame_interval=1,
# )
# animate_quiver(
#     loop_results_x[:, :, 0],
#     loop_results_x[:, :, 1],
#     loop_results_F[:,0],
#     loop_results_F[:,1],
#     arrow_scaling=0.01,
#     frame_interval=1,
# )