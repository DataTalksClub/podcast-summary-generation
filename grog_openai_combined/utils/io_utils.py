def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def save_text(content, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
