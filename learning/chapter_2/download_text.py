import os
import urllib.request


def main():

    if not os.path.exists("the-verdict.txt"):
        url = ("https://raw.githubusercontent.com/rasbt/"
               "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
               "the-verdict.txt")
        file_path = "the-verdict.txt"
        urllib.request.urlretrieve(url, file_path)


if __name__ == '__main__':
    main()
