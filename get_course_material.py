#! /usr/bin/env python3
import os
import sys

import urllib.request
file_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    download_hw_and_sol()
    download_data()
    print("downloads completed")

def download_hw_and_sol():
    baseurl = "https://work.caltech.edu/homework/"
    for i in range(1,9):
        hw_suffix = "hw{0}.pdf".format(i)
        hw_url = baseurl + hw_suffix
        ass_dir = os.path.join(file_dir, "ass" + str(i))
        download_file(hw_url, ass_dir, hw_suffix) # download hw to respective ass folder
        sol_suffix = "hw{0}_sol.pdf".format(i)
        sol_url = baseurl + sol_suffix
        download_file(sol_url, ass_dir, sol_suffix) # download hw_sol to respective ass folder 
    hw_suffix = "final.pdf"
    hw_url = baseurl + hw_suffix
    ass_dir = os.path.join(file_dir, "ass9")
    download_file(hw_url, ass_dir, hw_suffix)
    sol_suffix = "final_sol.pdf"
    sol_url = baseurl + sol_suffix
    download_file(sol_url, ass_dir, sol_suffix)
    
def download_data():
    # data for ass6
    ass_dir = os.path.join(file_dir, "ass6")
    in_url = "http://work.caltech.edu/data/in.dta"
    download_file(in_url, ass_dir, "in.dta")
    out_url = "http://work.caltech.edu/data/out.dta"
    download_file(out_url, ass_dir, "out.dta")
    # data for ass8
    ass_dir = os.path.join(file_dir, "ass8")
    features_train_url = "http://www.amlbook.com/data/zip/features.train"
    download_file(features_train_url, ass_dir, "features.train")
    features_test_url = "http://www.amlbook.com/data/zip/features.test"
    download_file(features_test_url, ass_dir, "features.test")
    # data for ass9
    ass_dir = os.path.join(file_dir, "ass9")
    features_train_url = "http://www.amlbook.com/data/zip/features.train"
    download_file(features_train_url, ass_dir, "features.train")
    features_test_url = "http://www.amlbook.com/data/zip/features.test"
    download_file(features_test_url, ass_dir, "features.test")
  
             

def download_file(download_url, dir_path, file_name):
    response = urllib.request.urlopen(download_url)
    file = open(os.path.join(dir_path, file_name), 'wb')
    file.write(response.read())
    file.close()
    print(file_name, "from", download_url)

if __name__ == "__main__":
    main()
