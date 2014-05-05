#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Display map of two cities to compare venues."""
import os
import pymongo
import flask as f
import cities as c
import ClosestNeighbor as cn
import FSCategories as fsc

app = f.Flask(__name__)
app.config.update(dict(
    DEBUG=True,
    MONGO_URL=os.environ.get('MONGOHQ_URL',
                             "mongodb://localhost:27017/foursquare"),
))
# set the secret key.  keep this really secret:
app.secret_key = os.environ['SECRET_KEY']
# TODO: find a better way to share complex object between requests
# http://pythonhosted.org/Flask-Cache/#flask.ext.cache.Cache.memoize
ORIGIN = {}
DEST = {}


def connect_db():
    """Return a client to the default mongo database."""
    return pymongo.MongoClient(app.config['MONGO_URL'])


def get_db():
    """Opens a new database connection if there is none yet for the current
    application context.
    """
    if not hasattr(f.g, 'mongo_db'):
        f.g.mongo_db = connect_db()
    return f.g.mongo_db


@app.teardown_appcontext
def close_db(error):
    """Closes the database again at the end of the f.request."""
    if hasattr(f.g, 'mongo_db'):
        f.g.mongo_db.close()


@app.route('/match', methods=['POST'])
def find_match():
    side = int(f.request.form['side'])
    assert side in [0, 1]
    _id = f.request.form['_id']
    first = DEST if side else ORIGIN
    second = ORIGIN if side else DEST
    query, res_ids, answers, dsts, among = cn.find_closest(_id, first, second)
    _ = cn.interpret(first['features'][query, :],
                     second['features'][answers[0], :])
    query_info, first_answer, feature_order = _
    answers_info = [first_answer]
    answers_info.extend([cn.interpret(first['features'][query, :],
                                      second['features'][answer, :],
                                      feature_order)[1]
                         for answer in answers[1:]])
    sendf = lambda x, p: ('{:.'+str(p)+'f}').format(float(x))
    res = {'query': query_info, 'answers_id': list(res_ids),
           'distances': [sendf(d, 5) for d in dsts],
           'explanations': answers_info, 'among': among}
    return f.jsonify(r=res)


@app.route('/populate', methods=['POST'])
def get_venues():
    origin = f.request.form['origin']
    vids = (ORIGIN if origin == "true" else DEST)['index']
    db = get_db().get_default_database()['venue']
    res = db.find({'_id': {'$in': vids}}, {'name': 1, 'cat': 1,
                                           'canonicalUrl': 1, 'loc': 1})
    ven = [{'name': v['name'], 'cat': fsc.CAT_TO_ID[:v['cat']],
            '_id': v['_id'], 'url': v['canonicalUrl'],
            'loc': list(reversed(v['loc']['coordinates']))} for v in res]
    return f.jsonify(r=ven)


@app.route('/<origin>/<dest>/<int:knn>')
def compare(origin, dest, knn):
    """Compare two cities."""
    global ORIGIN
    global DEST
    origin = 'barcelona' if origin not in c.SHORT_KEY else origin
    dest = 'helsinki' if dest not in c.SHORT_KEY else dest
    ORIGIN = cn.gather_info(origin, knn, raw_features=True)
    DEST = cn.gather_info(dest, knn, raw_features=True)
    return f.render_template('cnn.html', origin=origin, dest=dest, knn=knn,
                             lbbox=c.BBOXES[origin], rbbox=c.BBOXES[dest])


@app.route('/')
def welcome():
    return f.redirect(f.url_for('compare', origin='barcelona',
                                dest='helsinki', knn=2))

if __name__ == '__main__':
    app.run()
