import hashlib

def calculate_hashes(filename):
    BUF_SIZE = 65536  # Buffer size for reading the file in chunks

    md5 = hashlib.md5()
    sha256 = hashlib.sha256()

    with open(filename, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
            sha256.update(data)

    print("MD5: {0}".format(md5.hexdigest()))
    print("SHA256: {0}".format(sha256.hexdigest()))

while True:
    filename = input("Enter the file name whose hash you want to check: ")
    calculate_hashes(filename)
    
    choice = input("Do you want to check another file? (Y/N): ").strip().lower()
    if choice != 'y':
        print("Exiting the program.")
        break
