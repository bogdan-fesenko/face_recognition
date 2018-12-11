

import sqlalchemy
import pandas as pd
from datetime import datetime
import numpy as np
from collections import Counter


class DatabaseWorker:
    def __init__(self):
        print("\nStart connecting to a database...\n")
        self.distance_threshold = 0.6#800
        self.vote_between_n_closest_faces = 10  # we select n closest faces and select one most frequent among them
        self.path_to_imagesdb_folder = 'data/images/db/'

        db_hostname = 'localhost'
        db_port = '5432'
        db_name = 'facesdb'
        db_user = 'postgres'
        self.conn_string = 'postgresql://'+db_user+'@'+db_hostname+':'+db_port+'/'+db_name
        # self.conn = None
        self.engine = None

        self.create_connection()
        try:
            self.engine.execute("CREATE EXTENSION CUBE;")
        except:
            print("msg: DB: Cube extension already exist!")
        try:

            self.engine.execute("""
                CREATE TABLE if not exists faces(
                    id int NOT NULL,
                    name varchar(50) NOT NULL,
                    time timestamp NOT NULL,
                    color varchar(25) NOT NULL,
                    detection_confidence int,
                    embedding cube NOT NULL)            
                """)
        except Exception as ex:
            print("Error when creating table faces:", ex)
            # print("DB: Table faces already exist!")

        # self.close_connection()
        print("\nConnected to the database!\n")

        # self.create_connection()


    def create_connection(self):
        # self.conn = sqlalchemy.create_engine(self.conn_string).connect()
        self.engine = sqlalchemy.create_engine(self.conn_string)

    def close_connection(self):
        print("\nmsg: Connection to the DB was closed!\n")
        self.engine.dispose()
        # self.conn.close()

    #
    # def query_get_closest_id(self, face):
    #
    #     face_encoding_string = ",".join([str(x) for x in face.embedding])
    #     query_get_closest_id = """
    #                 SELECT
    #                     id, name, color
    #                 FROM faces
    #                 ORDER BY embedding <-> cube(array[{}])
    #                 LIMIT 1
    #                 """.format(face_encoding_string)
    #     try:
    #         is_faces_table_empty = False
    #         df_face_info_output = pd.read_sql_query(query_get_closest_id, self.engine).iloc[0]
    #         # print(df_face_info_output)
    #         face.id = df_face_info_output['id']
    #         face.name = df_face_info_output['name']
    #         face.color = df_face_info_output['color']
    #     except Exception as ex:
    #         print("msg: Error when finding info of accepted closes face:", ex)
    #         is_faces_table_empty = True
    #
    #     return face

    def get_distances(self, face):
        # self.create_connection()

        face_encoding_string = ",".join([str(x) for x in face.embedding])

        # query_get_closest_id = """
        #     SELECT
        #         id,
        #         name,
        #         color
        #     FROM faces
        #     ORDER BY embedding <-> cube(array[{}])
        #     LIMIT 1
        #     """.format(face_encoding_string)
        # try:
        #     is_faces_table_empty = False
        #     df_face_info_output = pd.read_sql_query(query_get_closest_id, self.engine).iloc[0]
        #     # print(df_face_info_output)
        #     face.id = df_face_info_output['id']
        #     face.name = df_face_info_output['name']
        #     face.color = df_face_info_output['color']
        # except:
        #     is_faces_table_empty = True


        # ----  find 10 closest distance below threshold in table
        query_get_closest_distance = """
            SELECT 
                id,
                name,
                color,
                embedding <-> cube(array[{0}]) AS distance
            FROM faces
            WHERE embedding <-> cube(array[{0}]) < {1}
            ORDER BY embedding <-> cube(array[{0}])
            LIMIT {2}
            """.format(face_encoding_string, str(self.distance_threshold), str(self.vote_between_n_closest_faces))
        try:
            #--(embedding <-> cube(array[{0}])) AS distance
            #embedding <-> cube(array[{0}]
            #ORDER BY embedding <-> cube(array[{0}])
            flag_find_distance_executed = True
            df_info_10__closest_faces = pd.read_sql_query(query_get_closest_distance, self.engine)
            list_id = list(df_info_10__closest_faces['id'])

            # choose answer
            if df_info_10__closest_faces['id'].nunique() <= int(self.vote_between_n_closest_faces/3):
                # if we have less than (10/3) unique nearest faces
                # select most frequent id
                face.id = sorted(list_id, key=Counter(list_id).get, reverse=True)[0]
            else:
                face.id = df_info_10__closest_faces['id'].iloc[0]  # choose closest id if we have more that 1/3 unique nearest faces

            [face.name, face.name, face.color, face.proba] = list(df_info_10__closest_faces[df_info_10__closest_faces['id']==face.id].iloc[0])
            face.proba = round(face.proba, 2)

            # distance = df_info_10__closest_faces.loc[0,"distance"]
            # print("distance", distance)
            # face.proba = round(distance, 2)
            # face
        except Exception as ex:
            print("msg: Error when finding closest distance in DB:", ex)
            flag_find_distance_executed = False

        # ------  if table is empty (one of possible err    or) or closest distance is above threshold
        if flag_find_distance_executed == False:# or distance >= self.distance_threshold:
            face = self.add_new_person(face)
            self.close_connection()
            return face


        # ------ find info about accepted face in DB
        # face = self.query_get_closest_id(face)  # TEST time of that query

        # self.close_connection()
        return face




    def add_new_person(self, face):
        # self.create_connection()

        face_encoding_string = ",".join([str(x) for x in face.embedding])

        query_find_max_id = """
            SELECT
                MAX(id) AS id
            FROM faces
            """

        max_id = pd.read_sql_query(query_find_max_id, self.engine).loc[0, "id"]
        if max_id == None:
            max_id = 0
        face.id = max_id + 1

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query_insert_new_person = """
            INSERT 
                INTO faces(id, name, time, color, detection_confidence, embedding)
            VALUES ({},'{}',TIMESTAMP '{}','{}',{},cube(array[{}]))
            """.format(str(face.id),
                       'unknown',
                       str(timestamp),
                       str(face.color),
                       str(face.detection_confidence*100),
                       face_encoding_string)

        self.engine.execute(query_insert_new_person)
        face.name = 'unknown'
        face.proba = 1.000

        print("New face inserted into DB!")

        # save image of recognized person
        np.save(self.path_to_imagesdb_folder+str(face.id)+'.npy', face.image)  # .npy extension is added if not given


        # self.close_connection()
        return face

# np.save('test3.npy', a)    # .npy extension is added if not given
# In  [9]: d = np.load('test3.npy')
# In [10]: a == d
# Out[10]: array([ True,  True,  True,  True], dtype=bool)


