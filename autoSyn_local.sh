#!/bin/sh
git add *  
git commit -am 'add some code from Mac'
# git commit -m 'add some results from Server'
git pull --rebase origin main   #domnload data
git push origin main            #upload data
