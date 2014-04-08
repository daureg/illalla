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
    DEBUG=os.environ.get('DEBUG', True),
    MONGO_URL=os.environ.get('MONGOHQ_URL',
                             "mongodb://localhost:27017/foursquare"),
))
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
    print(len(ven))
    return f.jsonify(r=ven)


@app.route('/<origin>/<dest>')
def compare(origin, dest):
    """Compare two cities."""
    global ORIGIN
    global DEST
    origin = 'barcelona' if not origin in c.SHORT_KEY else origin
    dest = 'helsinki' if not dest in c.SHORT_KEY else dest
    ORIGIN = ORIGIN or cn.gather_info(origin)
    DEST = DEST or cn.gather_info(dest)
    return f.render_template('draw.html', origin=origin, dest=dest,
                             lbbox=c.BBOXES[origin], rbbox=c.BBOXES[dest])


@app.route('/')
def welcome():
    return f.redirect(f.url_for('compare', origin='barcelona',
                                dest='helsinki'))


# set the secret key.  keep this really secret:
app.secret_key = os.environ['SECRET_KEY']
if __name__ == '__main__':
    app.run()
