from flask import Flask, render_template, request, redirect, url_for, send_file, current_app
from flask_sqlalchemy import SQLAlchemy
from app_utils import create_new_record, create_new_jupyter_notebook
from concurrent.futures import ThreadPoolExecutor
import os

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
    return render_template('ajax_table.html', jupyter_url='#', title='PyHealth OMOP (alpha)')

@app.route('/api/data')
def data():
    return {'data': [job.to_dict() for job in Job.query]}

@app.route('/create_job', methods=["GET", "POST"])
def create_job():
    new_jupyter_notebook_url = create_new_jupyter_notebook()
    return render_template('ajax_table.html', jupyter_url=new_jupyter_notebook_url, title='PyHealth OMOP (alpha)')

# def old_create_job():
#     output = request.form.to_dict()
#     config = {'dataset': output['dataset'], 'task': output['task'], 'model': output['model']} 
#     trigger a ML job
#     executor.submit(create_new_record, Job, db, config)
#     return redirect(url_for('home'))

@app.route('/download/<path:file_path>')
def downloadFile(file_path):
    path = os.path.join(current_app.root_path, file_path)
    print (path)
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=6789)