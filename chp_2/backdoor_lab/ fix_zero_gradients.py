
def fix_tuap_file():
    file_path = 'core/attacks/TUAP.py'

    # Read the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Replace the import
    old_import = 'from torch.autograd.gradcheck import zero_gradients'
    new_function = '''
def zero_gradients(x):
    if x.grad is not None:
        x.grad.zero_()
'''

    # Update content
    content = content.replace(old_import, new_function)

    # Write back
    with open(file_path, 'w') as file:
        file.write(content)

    print("TUAP.py has been updated!")

if __name__ == "__main__":
    fix_tuap_file()