from train_plot import figure1, compute_plot, load
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RNN on simulated data with FORCE')
    parser.add_argument('--function', type=str, choices=['triangle', 'square', 'periodic', 'complex_periodic', 'noisy_periodic', 'lorenz'], default='triangle')
    parser.add_argument('--simulation_duration', type=int, default=6000)
    parser.add_argument('--learning_start', type=int, default=2000)
    parser.add_argument('--learning_stop', type=int, default=4000)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--delta_t', type=float, default=0.2)
    parser.add_argument('--N_G', type=int, default=1500)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=10.0)
    parser.add_argument('--g_G_G', type=float, default=1.5)
    parser.add_argument('--p_G_G', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--import_data', type=bool, default=False)

    args = parser.parse_args()

    if args.import_data:
        data = load()
        figure1(data[0], data[1], data[2], data[3], data[4], data[5], data[6], t_max = 6000,
                learning_start = 2000, learning_stop = 4000, dt = 1)

    else:
        compute_plot(args.function, dt=1, delta_t=2, save=True, seed=args.seed)
        compute_plot(args.function, t_max=args.simulation_duration, learning_start=args.learning_start,
                     learning_stop=args.learning_stop, dt=args.dt, delta_t=args.delta_t,N_G=args.N_G, alpha=args.alpha,
                     tau=args.tau, g_G_G=args.tau, g_Gz=args.g_Gz, p_G_G=args.p_G_G, seed=args.seed, save=args.save)
