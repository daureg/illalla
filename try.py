#! /usr/bin/python2
# vim: set fileencoding=utf-8
import sys
sys.path.append('../spatialtopics')
from poly_gaussian import poly_to_gaussian, gaussian_to_poly
from timeit import default_timer as clock
import cities
# cities.WWW_CITIES = ['paris', 'salvador', 'sandiego']
from worldwide import rigged_query
import argparse
import json
import numpy as np
import os
import pickle
import persistent as p
from multiprocessing import Pool
NUM_THREADS = 6

paris_geo = { "type": "Polygon", "coordinates": [
    [ [ 2.3079442977905273, 48.86584400488787 ],
     [ 2.2887611389160156, 48.864771224562226 ],
     [ 2.2878599166870113, 48.85387273165656 ],
     [ 2.296614646911621, 48.84836611411784 ],
     [ 2.3166561126708984, 48.8542115806468 ],
     [ 2.3079442977905273, 48.86584400488787 ] ] ] }

salvador_geo = { "type": "Polygon", "coordinates": [ [
            [ -38.46914291381835, -12.9641614998626 ],
            [ -38.500986099243164, -12.965667067762352 ],
            [ -38.520469665527344, -12.998619992762045 ],
            [ -38.47394943237305, -13.007902895478635 ],
            [ -38.45987319946289, -12.984904077694585 ],
            [ -38.46914291381835, -12.9641614998626 ] ] ] }

def do_query(args):
    # source, target, region_s, region_t = args
    return rigged_query(*args)


if __name__ == '__main__':
    # print(rigged_query('paris', 'salvador', paris_geo, salvador_geo))

    parser = argparse.ArgumentParser()
    parser.add_argument('city', help='Name of query city')
    args = parser.parse_args()
    pool = Pool(NUM_THREADS)

    city_left = args.city
    others_cities = [_ for _ in cities.WWW_CITIES if _ != city_left]

    model_prefix_1 = "../spatialtopics/sandbox/" + city_left
    print("[Left] Loading Model ", model_prefix_1)
    m1 = pickle.load(open(model_prefix_1 + ".mdl", "rb"))
    scaler_1 = pickle.load(open(model_prefix_1 + ".scaler", "rb"))
    model_parameters_1 = m1.get_params()
    centers_1 = model_parameters_1.topic_centers
    covars_1 = model_parameters_1.topic_covar
    all_results = {}

    for city_right in others_cities:
        model_prefix_2 = "../spatialtopics/sandbox/" + city_right
        print("[Right] Loading Model ", model_prefix_2)
        m2 = pickle.load(open(model_prefix_2 + ".mdl", "rb"))
        scaler_2 = pickle.load(open(model_prefix_2 + ".scaler","rb"))
        model_parameters_2 = m2.get_params()
        centers_2 = model_parameters_2.topic_centers
        covars_2 = model_parameters_2.topic_covar
        all_results[city_right] = []

        diag = 'Gonna run {} queries from {} to {}'
        print(diag.format(len(centers_1), city_left, city_right))
        # go over product of regions and return a matrix of distances
        res = []
        for i, (center, cov) in enumerate(zip(centers_1, covars_1)):
            args = []
            # if i > NUM_THREADS:
            #     break
            poly_1 = gaussian_to_poly(center, cov, city_left, '', scaler_1,
                                      resolution=18)['geometry']
            for j, (center2, cov2) in enumerate(zip(centers_2, covars_2)):
                # if j > NUM_THREADS:
                #     break
                poly_2 = gaussian_to_poly(center2, cov2, city_left, '', scaler_2,
                                          resolution=18)['geometry']
                args.append((city_left, city_right, poly_1, poly_2))
            res.append(list(pool.map(do_query, args,
                                     chunksize=len(args)//NUM_THREADS)))
        all_results[city_right] = np.array(res)
    p.save_var('{}_rigged_EMD_matches.my'.format(city_left), all_results)
