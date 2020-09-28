import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
from werkzeug.utils import secure_filename
from lips_detect import *

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg' ,'.gif']
app.config['UPLOAD_PATH'] = './static/uploads'
app.config['LIPS_FOLDER'] = './static/LipsImage'

def remove_file(mydir):
    filelist = [ f for f in os.listdir(mydir)]
    print(filelist)
    for f in filelist:
        os.remove(os.path.join(mydir, f))
remove_file('./static/uploads')
remove_file('./static/LipsImage')

def set_imagepath(mydir):
    root =  os.getcwd()
    mydir = root+ mydir
    filelist = [ f for f in os.listdir(mydir)]
    img_dir = mydir+"/"+filelist[0]
    return img_dir

def validate_image(stream):
    header = stream.read(512)  # 512 bytes should be enough for a header check
    stream.seek(0)  # reset stream pointer
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files)

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    print("CALL THIS");
    hex_value = request.form['hexcode']
    filename = secure_filename(uploaded_file.filename)
    print(filename)
    if filename != '' and hex_value != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        remove_file('./static/uploads')
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        image_path = set_imagepath('/static/uploads')
        remove_file('./static/LipsImage')
        save_image(image_path, hex_value)
        print(hex_value, filename)

    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)




@app.route('/lips')
def lips():
    mydir = './static/LipsImage'
    filelist = [ f for f in os.listdir(mydir)]
    img_name = filelist[0]
    print(img_name)
    full_filename = os.path.join(app.config['LIPS_FOLDER'], img_name)
    return render_template('home_lips.html', user_image = full_filename)



# if __name__ == '__main__':
#     app.run(debug=True)
