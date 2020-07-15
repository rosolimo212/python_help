# доступ по ssh (по ключу)
ssh roman_vm@130.193.39.101

# ssh copy
scp hw.ipynb roman_vm@84.201.174.48:/home/roman_vm/

# превратить ноутбук в питоновский файл
jupyter nbconvert --to python hw.ipynb

# запуск на удалённого юпитера
# 1) remote machine:
python -m IPython notebook --no-browser --port=8889
# 2) own machine:
ssh -N -f -L localhost:8888:localhost:8889 roman_vm@130.193.39.101
# 3) пройти по ссылке с токеном

# перемещение разделов
sudo dd if=/dev/sda1 of=/media/roman/b175ada6-d71d-4403-9f4a-d2b663021f0d/backup/sda5_2020_07_05_1504.img

# postgres
# запустить консоль 
sudo -u postgres psql

# создать пользователя, базу, дать права
create user dev_test_user with password '';
create database third_db;
grant all privileges on database third_db to dev_test_user

# запуск консоли от имени
psql -h localhost third_db dev_test_user
# запуск запроса от имени
psql -h localhost third_db dev_test_user -c "SELECT table_schema,table_name FROM information_schema.tables"
