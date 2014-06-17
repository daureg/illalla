#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Read gold geometry from website DB."""

if __name__ == '__main__':
    # pylint: disable=C0103
    import sys
    import pymongo
    import ujson
    from collections import defaultdict
    from api_keys import MONGOHQ_URL
    uid = sys.argv[1]
    client = pymongo.MongoClient(MONGOHQ_URL)
    db = client.get_default_database()
    new_gold = defaultdict(list)
    target_city = ""
    for ans in db.answers.find({"uid": uid, "question": {"$exists": True}}):
        target_city = str(ans['city'])
        if 'geo' in ans:
            area = {"type": "Feature",
                    "properties": {"nb_venues": -1, "ref": uid}}
            is_circle = ans['type'] == 'circle'
            if is_circle:
                radius, center = ans['radius'], ans['geo']['coordinates']
                area["geometry"] = {"radius": radius, "center": center,
                                    "type": "circle"}
            else:
                area["geometry"] = ans['geo']
            new_gold[ans['question']].append(area)

    with open('static/cpresets.json') as infile:
        regions = ujson.load(infile)

    for district, info in regions.iteritems():
        if district not in new_gold:
            continue
        if target_city not in info['gold']:
            info['gold'][target_city] = []
        info['gold'][target_city].extend(new_gold[district])

    with open('static/cpresets.js', 'w') as out:
        out.write('var PRESETS =' + ujson.dumps(regions) + ';')
    with open('static/cpresets.json', 'w') as out:
        ujson.dump(regions, out)
