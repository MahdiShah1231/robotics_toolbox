import matplotlib.pyplot as plt


def draw_environment(workspace_width = 950, workspace_height = 950, robot_base_radius = 100):
    ## GIVE PARAMS IN MM
    vertices = [(0, -robot_base_radius),
                (0, workspace_height - robot_base_radius),
                (workspace_width, workspace_height - robot_base_radius),
                (workspace_width, -robot_base_radius)]

    plt.plot(*zip(*vertices), 'ko--')
    obstacle = plt.Circle((530, 300), 75, color="black")
    plt.gcf().gca().add_artist(obstacle)

if __name__ == '__main__':
    draw_environment()
    plt.axis("image")
    plt.show()