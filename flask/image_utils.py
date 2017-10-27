"""
Create an image from the result
"""
import requests
from bs4 import BeautifulSoup as BS
import subprocess
import os
import uuid

url = 'http://www.cs.toronto.edu/~graves/handwriting.cgi?text={}&style=..%2Fdata%2Ftrainset_diff_no_start_all_labels.nc%2C1082%2B554&bias=0.6&samples=1'

parent_path = os.path.abspath(os.path.curdir)
full_path = os.path.join(parent_path, 'public')

js_script = """
var page = require('webpage').create();
console.log('Setting content');
page.content = `{}`;
console.log('Render...');
page.render('{}');
console.log('All done!');
phantom.exit();
"""

HTML = """
<html>
<body>
{}
<br>
{}
</body>
</html>
"""

def download_images(text):
    global url
    result = requests.get(url=url.format(text))

    if result.status_code != 200:
        return None

    # Parse the result
    page = BS(result.text, 'html5lib')

    images = list(page.findAll('img'))

    # Sanity
    if len(images) < 7:
        return None

    # This will be the first image from
    # the page which is of interest
    image = images[6]

    return image


def render_image(html):
    global full_path, js_script

    file_name = "{}.png".format(os.path.join(full_path, str(uuid.uuid4())))
    script_path = os.path.join(full_path, 'generate_image.js')
    phantom_path = ".{}".format(os.path.join(parent_path, 'phantomjs'))
    print("file_path:", file_name)
    print("Script path:", script_path)
    print("Phantom path:", phantom_path)

    with open(script_path, 'w') as f:
        f.write(js_script.format(html, file_name))

    try:
        result = subprocess.check_call(['./phantomjs', script_path])
        print("render result:", result)
    except Exception as e:
        print("error:", e)
    return file_name

def generate_html(question, answer):
    global HTML
    q_image = download_images(question.lower())
    a_image = download_images(answer.lower())

    if q_image is None or a_image is None:
        return None

    return HTML.format(q_image, a_image)
    

def create_image(question, answer):
    html = generate_html(question, answer)
    return render_image(html)


if __name__ == '__main__':
    with open('DERP.html') as f:
        res = render_image(f.read())
        print("RES:", res)