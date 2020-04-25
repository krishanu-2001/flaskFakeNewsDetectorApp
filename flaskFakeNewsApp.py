from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import grub3

app = Flask(__name__,static_url_path='/C:/Users/krishanu/Desktop/krishanupy/python/tf1.0/static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db  = SQLAlchemy(app)


class Todo(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    content = db.Column(db.String(200),nullable = False)
    completed = db.Column(db.Integer, default=0)
   # marking = db.Column(db.Integer, default=0)
    date_created = db.Column(db.DateTime, default = datetime.utcnow)


    def __repr__(self):
        return '<Task %r>' % self.id
db.create_all()
@app.route('/',methods = ['POST','GET'])
def index():
    if request.method == "POST":
        task_content = request.form['content']
        new_task = Todo(content = task_content)

        try:
            db.session.add(new_task)
            db.session.commit()
            return redirect('/')
        except:
            return 'an error occured creating task'

    else:
        tasks = Todo.query.order_by(Todo.date_created).all()
        return render_template('index.html', tasks = tasks)

@app.route('/delete/<int:id>')
def delete(id):
    task_to_delete = Todo.query.get_or_404(id)

    try :
        db.session.delete(task_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'there was a problem detecting that task'


@app.route('/update/<int:id>', methods=['GET', 'POST'])
def update(id):
    task = Todo.query.get_or_404(id)

    if request.method == 'POST':
        task.content = request.form['content']

        try:
            db.session.commit()
            return redirect('/')
        except:
            return 'there was an error updating your report'
    else:
        return render_template('update.html', task = task)

global_list = []
@app.route('/ml', methods=['GET','POST'])
def ml():
    if request.method == "POST":
        #getting meer string
        meer = request.form.get('txt_clf')
        meercat = grub3.string_result( grub3.analyse_text(classifier,vectorizer,meer))
        print(meercat)
        global_list.append([meer, meercat])
        print('kjkj')
        #return '<h1>{}</h1>'.format(meercat)
        return render_template('ml.html', meercat = meercat)
    else:
        print('hhhh')
        return render_template('ml.html')


@app.route('/ppe')
def ppe():
    #if request.method == "GET":

    print(global_list)
    return render_template('ml.html', global_list = global_list)

if __name__ == "__main__":
    training_data, evaluation_data = grub3.preprocessing_step()
    # print(training_data)
    vectorizer = TfidfVectorizer(binary='true')
    classifier = grub3.training_step(training_data, vectorizer)
    global_list = []
    app.run(debug=True)