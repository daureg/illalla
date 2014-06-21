#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Read gold geometry from website DB."""


def find_users(db_handle):
    """Find a list of user id and the city they have contributed to"""
    import datetime as dt
    start = dt.datetime(2014, 6, 17, 9)
    res = db_handle.answers.find({'when': {'$gte': start},
                                  'question': {'$exists': True}},
                                 {'uid': 1, 'city': 1, 'when': 1})
    final = set()
    for question in res:
        formatted = "\t\t".join([str(question['uid']), str(question['city']),
                                 question['when'].strftime('%Y-%m-%d_%Hh')])
        final.update([formatted])
    for _ in final:
        print(_)

if __name__ == '__main__':
    # pylint: disable=C0103
    import sys
    import pymongo
    import json
    from collections import defaultdict
    from api_keys import MONGOHQ_URL
    client = pymongo.MongoClient(MONGOHQ_URL)
    db = client.get_default_database()
    if len(sys.argv) <= 1:
        find_users(db)
        sys.exit()
    uid = sys.argv[1]
    all_city_new_gold = defaultdict(lambda: defaultdict(list))
    for ans in db.answers.find({"uid": uid, "question": {"$exists": True},
                                "geo": {"$exists": True}}):
        target_city = str(ans['city'])
        area = {"type": "Feature",
                "properties": {"venues": [], "ref": uid}}
        is_circle = ans['type'] == 'circle'
        if is_circle:
            radius, center = ans['radius'], ans['geo']['coordinates']
            area["geometry"] = {"radius": radius, "center": center,
                                "type": "circle"}
        else:
            area["geometry"] = ans['geo']
        all_city_new_gold[target_city][ans['question']].append(area)

    with open('static/ground_truth.json') as infile:
        regions = json.load(infile)

    for district, info in regions.iteritems():
        for target_city, new_gold in all_city_new_gold.iteritems():
            if district not in new_gold:
                continue
            if target_city not in info['gold']:
                info['gold'][target_city] = []
            info['gold'][target_city].extend(new_gold[district])

    # with open('static/cpresets.js', 'w') as out:
    #     out.write('var PRESETS =' + json.dumps(regions) + ';')
    with open('static/ground_truth.json', 'w') as out:
        json.dump(regions, out, sort_keys=True, indent=2,
                  separators=(',', ': '))
