import os

# Define the filename and content of the new script
filename = "my_script.sh"
content = "#!/bin/bash\n\necho 'Hello, World!'"

# Create the new file and write the content to it
with open(filename, "w") as f:
    f.write(content)

# Set the file permissions to allow execution
os.chmod(filename, 0o755)