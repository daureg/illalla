#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Plot time pattern of all cities."""
import matplotlib
matplotlib.use('Agg')
import VenueFeature as vf
import explore as xp
import numpy as np
import scipy.io as sio

DO_CLUSTER = lambda val, k: vf.cluster.kmeans2(val, k, 25, minit='points')


def plot_city(city, weekly=False, clusters=5):
    """Plot the 5 time clusters of `city` and save them on disk."""
    shift = 2  # start from 1am instead of midnight
    chunk = 4
    venue_visits = xp.get_visits(CLIENT, xp.Entity.venue, city)
    # Compute aggregated frequency for venues with at least 5 visits
    enough = {k: xp.to_frequency(xp.aggregate_visits(v, shift, chunk)[int(weekly)])
              for k, v in venue_visits.iteritems() if len(v) > 5}
    sval = np.array(enough.values())
    num_cluster = clusters
    min_disto = 1e9
    for _ in range(7):
        tak, tkl = DO_CLUSTER(sval, num_cluster)
        current_disto = vf.get_distorsion(tak, tkl, sval)
        if current_disto < min_disto:
            min_disto, ak, kl = current_disto, tak, tkl
    std_ord = np.argsort((np.argsort(ak)), 0)[:, -1]
    # vf.draw_classes(ak[std_ord, :], shift, chunk)
    # vf.plt.title('{}, {} venues'.format(city, len(enough)))
    # vf.plt.ylim([0, 0.28 if weekly else 0.9])
    city = 'times/' + city
    city += '_weekly' if weekly else '_daily'
    sio.savemat(city+'_time', {'t': ak[std_ord, :]}, do_compression=True)
    # vf.plt.savefig(city+'_time.png', dpi=160, transparent=False, frameon=False,
    #                bbox_inches='tight', pad_inches=0.1)
    # vf.plt.clf()

if __name__ == '__main__':
    # pylint: disable=C0103
    import arguments
    import persistent as p
    args = arguments.get_parser().parse_args()
    DB, CLIENT = xp.cm.connect_to_db('foursquare', args.host, args.port)
    res = {}
    for city in reversed(xp.cm.cities.SHORT_KEY):
        print(city)
        plot_city(city, weekly=False, clusters=5)
        # plot_city(ciy, weekly=True, clusters=)
    #     venue_visits = xp.get_visits(CLIENT, xp.Entity.venue, city)
    #     res.update({k: len(v) for k, v in venue_visits.iteritems()})
    # p.save_var('venue_visits', res)
