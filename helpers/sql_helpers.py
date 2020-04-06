import sys
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
    pass


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

def insert_d(s, source, date_crawled, date_created, malware_id, cnx, ioc_type, value):
    cursor = cnx.cursor()

    # Replace unwanted/error-inducing characters
    s = s.replace('\n', ' ')
    s = s.replace('\'', '')
    s = s.replace('\"', '')
    # s = s.replace('\'', '\'\'')
    print("NOW INSERTING DATA...")

    # * SELECT signature id
    cursor.execute(f"SELECT s_id from signatures WHERE ioc_type='{ioc_type}' AND value='{value}' AND malwares_m_id='{malware_id}';")
    result = cursor.fetchall()
    s_id = result[0][0]
    print("TYPE:", ioc_type, "VALUE:", value, "ID:", malware_id, "SIGNATURE ID:", s_id)
    print("PARA:", s, "M ID:", malware_id, "S ID:", s_id)

    # * Check if data already exists
    # Using EXISTS Condition is very inefficient as the sub-query is RE-RUN for EVERY row in the outer query's table. There are more efficient ways to write most queries
    cursor.execute(f"SELECT COUNT(*) FROM data WHERE text='{s}' AND malwares_m_id='{malware_id}' AND signatures_s_id='{s_id}';")
    result = cursor.fetchall()
    count = result[0][0]

    if count == 0:
        # if math.isnan(date_created)                 
        if not date_created:
            try:
                print("INSERTING NULL DATE")
                cursor.execute(f"INSERT INTO data (text, source, date_crawled, date_created, malwares_m_id, signatures_s_id) VALUES ('{s}', '{source}', '{date_crawled}', NULL, '{malware_id}', '{s_id}');")
                print("NULL SUCCESS")
            except Exception as e:
                print('FAILED TO INSERT WITH NULL DATE:', e)
        else:
            try:
                print("INSERTING DATA WITH DATE_CREATED", date_created)
                cursor.execute(f"INSERT INTO data (text, source, date_crawled, date_created, malwares_m_id, signatures_s_id) VALUES ('{s}', '{source}', '{date_crawled}', '{date_created}', '{malware_id}', '{s_id}');")
                print("SUCCESS")
            except Exception as e:
                print("FAILED TO INSERT WITH DATE CREATED:", e)
    else:
        print("THIS DATA ALREADY EXISTS, CARRY ON")
        
    # * Try committing changes to the database
    try:    
        cnx.commit()          
        print('COMMITTED')
    except Exception as e:
        print("COULD NOT COMMIT DATA:", e)
  
    cursor.close()
    cnx.close()
