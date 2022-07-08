from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from app_utils import create_new_record
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(10)

app = Flask(__name__)

# build SQL db
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# table format
class Job(db.Model):
    run_id = db.Column(db.Integer, primary_key=True)
    trigger_time = db.Column(db.DateTime, index=True)
    dataset = db.Column(db.String(50), index=True)
    task_name = db.Column(db.String(50))
    model = db.Column(db.String(50))
    run_stats = db.Column(db.String(256))
    downloads = db.Column(db.String(50))

    def to_dict(self):
        return {
            'run_id': self.run_id,
            'trigger_time': self.trigger_time,
            'dataset': self.dataset,
            'task_name': self.task_name,
            'model': self.model,
            'run_stats': self.run_stats,
            'downloads': self.downloads,
        }

# create db tables in db.sqlite
db.create_all()

@app.route("/")
@app.route("/home", methods=["GET", "POST"])
def home():
    return render_template('ajax_table.html', title='PyHealth OMOP (alpha)')

@app.route('/api/data')
def data():
    return {'data': [job.to_dict() for job in Job.query]}

@app.route('/create_job', methods=["GET", "POST"])
def create_job():
    output = request.form.to_dict()
    config = {'dataset': output['dataset'], 'task': output['task'], 'model': output['model']} 
    # trigger a ML job
    executor.submit(create_new_record, Job, db, config)
    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=6789)