#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Plot time pattern of all cities."""
import matplotlib
matplotlib.use('Agg')
import VenueFeature as vf
import explore as xp
import numpy as np
import scipy.io as sio

DO_CLUSTER = lambda val, k: vf.cluster.kmeans2(val, k, 20, minit='points')


def plot_city(city):
    """Plot the 5 time clusters of `city` and save them on disk."""
    daily = 0  # look a time of the day instead of day of the week
    shift = 1  # start from 1am instead of midnight
    venue_visits = xp.get_visits(CLIENT, xp.Entity.venue, city)
    # Compute aggregated frequency for venues with at least 5 visits
    enough = {k: xp.to_frequency(xp.aggregate_visits(v, shift)[daily])
              for k, v in venue_visits.iteritems() if len(v) > 5}
    sval = np.array(enough.values())
    num_cluster = 5
    min_disto = 1e9
    for _ in range(5):
        tak, tkl = DO_CLUSTER(sval, num_cluster)
        current_disto = vf.get_distorsion(tak, tkl, sval)
        if current_disto < min_disto:
            min_disto, ak, kl = current_disto, tak, tkl
    std_ord = np.argsort((np.argsort(ak)), 0)[:, -1]
    vf.draw_classes(ak[std_ord, :], shift)
    vf.plt.title('{}, {} venues'.format(city, len(enough)))
    vf.plt.ylim([0, 0.85])
    city = 'time/' + city
    sio.savemat(city+'_time', {'t': ak[std_ord, :]}, do_compression=True)
    vf.plt.savefig(city+'_time.png', dpi=160, transparent=False, frameon=False,
                   bbox_inches='tight', pad_inches=0.1)
    vf.plt.clf()

if __name__ == '__main__':
    # pylint: disable=C0103
    import arguments
    args = arguments.get_parser().parse_args()
    DB, CLIENT = xp.cm.connect_to_db('foursquare', args.host, args.port)
    for city in reversed(xp.cm.cities.SHORT_KEY):
        print(city)
        plot_city(city)
