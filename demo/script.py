from weasyprint import HTML
from pathlib import Path

# Sample content to replicate the design using Ubuntu font (Google Fonts)
title = "Building A Podcast Summary Generator"
guest_name = "Alexey Grigorev"
tag_label = "PODCAST"

# Path to the microphone icon (SVG format) and guest image
microphone_icon_path = Path("../resources/icons/microphone.svg").resolve().as_uri()
guest_image_path = Path("../resources/images/front-page/alexeygrigorev.jpg").resolve().as_uri()


# This can be imported as a base HTML CSS template
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Ubuntu:wght@400;700&display=swap');
        @page {{
            size: A4 landscape;
            margin: 40px;
        }}
        body {{
            font-family: 'Ubuntu', sans-serif;
            margin: 0;
            padding: 40px;
            background-color: #f9f9f9;
            background-image: radial-gradient(#f1f1f1 1px, transparent 1px);
            background-size: 12px 12px;
        }}
        .tag-container {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-left: 10px;
            margin-top: 18px;
            margin-bottom: 25px;
                    }}
        .tag-icon {{
            background-color: #f9f9f9;
            width: 80px;
            height: 75x;
        }}
        .tag-title {{
            color: white;
            background-color: #0057B7;
            border-radius: 8px;
            font-size: 30px;
            padding: 6px 6px;
            font-weight: bold;
            margin-top: 18px;
            margin-bottom: 20px;
        }}
        .main {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-left: 10px;

        }}
        .title {{
            font-size: 55px;
            font-weight: 700;
            color: #0057B7;
            margin-left: 25px;
            width: 100%;
            max-width: 550px;
            word-wrap: break-word;
            white-space: normal;
        }}
        .guest-section {{
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;  /* ⬅️ Needed for absolute positioning inside */
        }}

        .guest-frame {{
            width: 270px;
            height: 270px;
            border-radius: 50%;
            border: 5px solid #0057B7;
            padding: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #f9f9f9;
            position: relative;  /* ⬅️ In case you want to position name relative to this instead */
        }}

        .guest-photo {{
            border-radius: 100%;
            width: 250px;
            height: 250px;
            background-image: url('{guest_image_path}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .guest-name {{
            background-color: #0057B7;
            color: white;
            font-weight: bold;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 28px;
            position: absolute;
            bottom: -20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 2;
            white-space: nowrap;  /* 👈 prevents line breaks */
        }}


    </style>
</head>
<body>
    <div class="tag-container">
        <img class="tag-icon" src="{microphone_icon_path}"  />
        <div class="tag-title">{tag_label}</div>
    </div>
    <div class="main">
        <div class="title">{title}</div>
        <div class="guest-section">
            <div class="guest-frame">
                <div class="guest-photo"></div>
            </div>
            <div class="guest-name">{guest_name}</div>
        </div>
    </div>
</body>
</html>
"""


# Generate the PDF
output_file = "output/weasyprint_front_page.pdf"

# Create the output directory if it doesn't exist
Path(output_file).parent.mkdir(parents=True, exist_ok=True)
HTML(string=html_content).write_pdf(output_file)

if __name__ == "__main__":
    # Print the path of the generated PDF
    print(f"PDF generated successfully: {output_file}")
