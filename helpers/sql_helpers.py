import sys
from datetime import datetime
import mysql.connector
from mysql.connector import errorcode

            
def connect_db():
    # Attempt to connect to db
    try:
        cnx = mysql.connector.connect(user='malware', password='malware',
                                    host='10.255.252.177',
                                    database='malwareanalysis')
        print("CONNECTION SUCCESS", cnx)

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
            print(sys.stderr, "does not exist")
        else:
            print(err)
            print(sys.stderr, "does not exist")
        sys.exit(1)

    return cnx

def insert_data(cnx, attack_tid, name, sentence):
    cursor = cnx.cursor()
    date_crawled = datetime.today().strftime('%Y-%m-%d')

    # * Try inserting data
    try:
        print("INSERTING INTO DATABASE")
        cursor.execute(f"INSERT INTO tram (attack_tid, t_name, date_crawled) VALUES ('{attack_tid}', '{name}', '{date_crawled}';")
        print("INSERT SUCCESS")
    except Exception as e:
        print('FAILED TO INSERT WITH NULL DATE:', e)

    # * Try committing changes to the database
    try:    
        cnx.commit()          
        print('COMMITTED')
    except Exception as e:
        print("COULD NOT COMMIT DATA:", e)

    cursor.close()
    cnx.close()

def insert_signature(ioc_type, value, malware_id, cnx):
    cursor = cnx.cursor()  

    # Note: Try catch does not work as cursor just returns a None object and no error is returned
    cursor.execute(f"SELECT COUNT(*) from signatures WHERE ioc_type='{ioc_type}' AND value='{value}' AND malwares_m_id='{malware_id}';")
    result = cursor.fetchall()
    counts = result[0][0]

    # * If the signature does not already exist, insert into db
    if counts == 0:
        cursor.execute(f"INSERT INTO signatures (ioc_type, value, malwares_m_id) VALUES ('{ioc_type}', '{value}', '{malware_id}');")   
        print("INSERTING")

        # * Try committing changes to the database
        try:    
            cnx.commit()          
            print('COMMITTED')
        except Exception as e:
            print("COULD NOT INSERT DATA:", e) 
    else:
        print("ALREADY EXISTS")

    cursor.close()
    cnx.close()