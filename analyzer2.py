import os
import ast
import glob

target_dirs = [
    'app/ai/agents',
    'app/ai/tools',
    'app/api',
    'app/core',
    'app/schemas',
    'app/ai/providers',
    'app/ai/memory'
]

specific_files = [
    'app/config.py',
    'app/main.py',
    'app/ai/enums.py',
    'app/db/models.py',
    '.env'
]

md_files = glob.glob('*.md')

all_files = specific_files + md_files

for root, dirs, files in os.walk('app'):
    root_f = root.replace(os.sep, '/')
    if any(root_f.startswith(d) for d in target_dirs):
        for f in files:
            if f.endswith('.py'):
                all_files.append(os.path.join(root, f))

# filter out duplicates and normalize paths
all_files = list(set([os.path.normpath(f).replace('\\', '/') for f in all_files]))

report = []

for filepath in all_files:
    if not os.path.exists(filepath):
        continue
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        continue

    file_report = f"## File: {filepath}\n"
    
    if filepath.endswith('.md') or filepath.endswith('.env'):
        # Just give a short summary of text files
        lines = content.split('\n')
        file_report += f"- **Purpose:** Configuration/Documentation file. ({len(lines)} lines)\n"
        if filepath.endswith('.md'):
            file_report += f"- **Content Summary:**\n"
            file_report += "\n".join([f"  > {l}" for l in lines[:5]]) + ("\n  > ..." if len(lines)>5 else "") + "\n"
        elif filepath.endswith('.env'):
            keys = [l.split('=')[0] for l in lines if '=' in l and not l.strip().startswith('#')]
            file_report += f"- **Env Keys:** {', '.join(keys)}\n"
        report.append(file_report)
        continue

    # Parse python
    try:
        tree = ast.parse(content)
    except Exception:
        file_report += "- **Purpose:** Failed to parse AST.\n"
        report.append(file_report)
        continue

    docstring = ast.get_docstring(tree)
    if docstring:
        file_report += f"- **Purpose:** {docstring.split(chr(10))[0]}\n"
    else:
        file_report += "- **Purpose:** No module docstring.\n"
        
    classes = []
    functions = []
    imports = []
    todos = []
    
    for i, line in enumerate(content.split('\n')):
        if 'TODO' in line or 'pass' in line or 'NotImplementedError' in line or '...' in line:
            if line.strip().startswith('#') or 'pass' in line or 'NotImplementedError' in line:
                todos.append(f"Line {i+1}: {line.strip()}")
                
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            functions.append(node.name)

    file_report += f"- **Dependencies/Imports:** {', '.join(set(imports))}\n"
    file_report += f"- **Key Classes:** {', '.join(classes)}\n"
    file_report += f"- **Key Functions:** {', '.join(functions)}\n"
    
    implemented_status = "Fully implemented" if not todos else "Contains stubs/TODOs"
    file_report += f"- **Status:** {implemented_status}\n"
    if todos:
        file_report += f"- **Stubs/TODOs found:** {len(todos)} instances (e.g. {todos[0][:40]})\n"

    report.append(file_report)

with open('full_report2.md', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(report))
