from tkinter import *
from tkinter import filedialog
import pandas as pd
import os
import threading
from fuzzywuzzy import fuzz, process
from pathlib import Path
from tkinter import messagebox
from tkinter import PhotoImage
import pickle
import datetime
import shutil


counter =0
counter1=0
class leap():

    def __init__(self,master):
        self.abc= 0
        root.title("Legal Entity Automation Process Tool")
        root.iconbitmap(r'EY.ico')
        root.configure(background= 'white')
        root.geometry("420x475")
        root.resizable(width=False, height=False)
        #root.resizable(width=True, height=True)
        logo = PhotoImage(file = os.getcwd()+r'\EY.gif')

        label_logo = Label(root,bg='white',image=logo)
        label_logo.image = logo
        label_logo.grid(row =2,column=0, columnspan = 2, rowspan=1, sticky=N+E+S+W)
        label_root = Label(root, bg='white', text='LEAP', font='Calibri 30 bold').grid(row=1,columnspan=2, rowspan=1,sticky=N + E + S + W)
        frame = Frame(master)
        frame.place(x = 10, y = 5,width=100,height=100)
        frame.grid(column=1, row=4, sticky=E,pady=(0,15))

        self.file1_column = StringVar()
        self.file2_column = StringVar()
        self.perform_fuzz = StringVar()
        self.v= IntVar()
        self.var= IntVar()
        self.v1= IntVar()
        self.f1= IntVar()
        #Frame1 = Frame(root)
        #Frame1.grid()
        self.button = Button(root, text="Browse Data File", command=self.open_file1).grid(column=1, row=3, sticky=E,padx=4,pady=(0,10))
        datafile_label= Label(root,bg='white',text='**Input datafile in desired format', font='Calibri 8').grid(row=3, sticky=W,padx=4,pady=(0, 10))
        #self.button = Button(Frame1, text="Browse Data File", command=self.open_file1).grid(column=1, row=3, sticky=E,padx=4, pady=(0, 10))
        self.s1=Scrollbar(frame,orient='vertical')
        #s1.pack(side='right', fill='y')
        self.s1.grid(column=1, row=4, sticky=E)
        self.lb = Listbox(frame, width=30, height=5, selectmode=MULTIPLE, yscrollcommand=self.s1.set,bg='white')
        self.lb.grid(column=1, row=4, sticky=E)
        self.lb.configure(yscrollcommand=self.s1.set)
        self.s1.config(command=self.lb.yview)
        listbox_label = Label(root, bg='white', text='Select Amount Columns', font='Calibri 10 bold').grid(row=4, sticky=W,padx=4,pady=(0, 15))
        file1_label = Label(root,bg = 'white',text='Enter keywords column from Data File',font='Calibri 10 bold').grid(row=11, sticky=W,padx=4,pady=(0, 15))
        file1_entry = Entry(root, textvariable = self.file1_column)
        file1_entry.columnconfigure(0,weight=1)
        file1_entry.grid(column = 1, row = 11, sticky = W+E, padx=4,pady=(0,15))

        frm_select_model = Frame(root,bg ='white')
        file_select_label = Label(root, bg='white', text='Select Solution Model', font='Calibri 10 bold').grid(row=5,sticky=W,padx=4,pady=(0, 15))
        self.rd_fuzz = Radiobutton(frm_select_model, text="Fuzzy Match", variable=self.v1, value=1, bg="white",command=self.disable_enable)
        self.rd_fuzz.grid(row=5, column=1, sticky=E, padx=4)
        self.rd_ml = Radiobutton(frm_select_model, text="ML Model", variable=self.v1, value=2, bg="white",command=self.disable_enable)
        self.v1.set(1)
        self.rd_ml.grid(row=5, column=2, sticky=E, padx=4)
        frm_select_model.place(x=0,y=6,width=100, height=100)
        frm_select_model.grid(column=1, row=5, sticky=E, pady=(0, 10))

        self.frm_file_type = Frame(root,bg ='white')
        self.file_type_label= Label(root, bg='white', text='Select File Type BS/PL', font ='Calibri 10 bold')
        self.file_type_label.grid(row=6,sticky=W,padx=4,pady=(0, 15))
        self.rd_file_BS= Radiobutton(self.frm_file_type,text='BS', variable= self.f1,value=1, bg='white',command = self.disable_enable)
        self.rd_file_BS.grid(row=6, column=1, sticky=E, padx=4)
        self.rd_file_PL= Radiobutton(self.frm_file_type,text='PL',variable= self.f1, value=2,bg='white',command = self.disable_enable)
        self.rd_file_PL.grid(row=6, column=2, sticky=E, padx=4)
        self.frm_file_type.place(x=0,y=6,width=100, height=100)
        self.frm_file_type.grid(column=1, row=6, sticky=E, pady=(0, 10))
        self.f1.set(1)

        self.label_fuzz= Label(root,text='Browse mapping file for Fuzzy match', font ='Calibri 10 bold',bg='white')
        self.label_fuzz.grid(row=7,sticky=W,padx=4,pady=(0, 15))
        #self.label_entry = Entry(root,bg='white').grid(column = 0,row=6, sticky=W)
        self.button_fuzz = Button(root, text= 'Browse Mapping File',command=self.open_mapping_file)
        self.button_fuzz.grid(row=7,column=1, sticky=E,padx=4,pady=(0,10))

        self.frm_model=Frame(root,bg ='white')
        self.file1_select_model = Label(root, bg='white', text='Select Model BS or PL', font='Calibri 10 bold')
        #self.file1_select_model.grid(row=7, sticky=W)
        self.rd_bs = Radiobutton(self.frm_model, text="BS", variable=self.var, value=1, bg="white", command=self.disable_enable)
        self.rd_bs.grid(row=7, column=1, sticky=E, padx=4)
        self.rd1_pl = Radiobutton(self.frm_model, text="PL", variable=self.var, value=2, bg="white", command=self.disable_enable)
        self.rd1_pl.grid(row=7, column=2, sticky=E, padx=4)
        self.frm_model.place(x=0,y=7,width=100, height=100)
        self.var.set(1)
        #self.frm_model.grid(column=1, row=7, sticky=E, pady=(0, 10))

        self.frm_radio= Frame(root, bg='white')
        self.file1_label_retrain = Label(root, bg='white', text='Do you want to re-train the model for BS/PL', font='Calibri 10 bold')
        #self.file1_label_retrain.grid(row=8, sticky=W)
        self.rd_yes = Radiobutton(self.frm_radio, text="Yes", variable=self.v, value=1, bg="white", command = self.disable_enable)
        #self.rd_yes.grid(row=8,column=1,sticky = E, padx =4)
        self.rd1_no =Radiobutton(self.frm_radio, text="No", variable=self.v, value=2, bg="white", command = self.disable_enable)
        #self.rd1_no.grid(row= 8,column=2,sticky = E, padx=4)
        self.v.set(2)
        self.frm_radio.place(x=0, y=7, width=100, height=100)
        #self.frm_radio.grid(column=1, row=8, sticky=E, pady=(0, 10))

        self.button1 = Button(root, text='Browse Training File',command=self.open_file2, state= DISABLED)
        self.training_file_label= Label(root,text='**Input training file in specified format', bg='white', font='calibri 8')
        #self.training_file_label.grid(row=9,sticky=W,padx=4,pady=(0, 10))
        #self.button1.grid(column=1,row=9,sticky=E,pady=(0, 15),padx=4)

        self.button_retrain= Button(root, text= 'Re-Train Model', command= self.retrain, state= DISABLED)
        self.retrain_label= Label(root,text='**Retraining will create a new .pkl file', bg='white', font='calibri 8')
        #self.retrain_label.grid(row=10,sticky=W,padx=4,pady=(0, 10))
        #self.button_retrain.grid(column = 1, row =10, sticky = E, pady=(0, 15),padx=4)

        self.button = Button(root, text='RUN', command=self.predict).grid(row=13,columnspan=2,pady=(0, 10),padx=4)

    def disable_enable(self):
        self.hide()
        self.unhide()
        self.button1.config(state= DISABLED if self.v.get()==2 else NORMAL)
        self.button_retrain.config(state= DISABLED if self.v.get()==2 else NORMAL)
        self.rd_bs.config(state= DISABLED if self.v1.get()==1 else NORMAL)
        self.rd1_pl.config(state= DISABLED if self.v1.get()==1 else NORMAL)
        self.rd1_no.config(state= DISABLED if self.v1.get()==1 else NORMAL)
        self.rd_yes.config(state= DISABLED if self.v1.get()==1 else NORMAL)

    def hide(self):
        if self.v1.get() == 1:
            root.geometry("440x450")
            self.frm_model.grid_remove()
            self.frm_radio.grid_remove()
            self.button_retrain.grid_remove()
            self.button1.grid_remove()
            self.file1_label_retrain.grid_remove()
            self.file1_select_model.grid_remove()
            self.training_file_label.grid_remove()
            self.retrain_label.grid_remove()

        else:
            self.button_fuzz.grid_remove()
            self.label_fuzz.grid_remove()
            self.rd_file_PL.grid_remove()
            self.rd_file_BS.grid_remove()
            self.file_type_label.grid_remove()
            self.frm_file_type.grid_remove()


    def unhide(self):
        if self.v1.get()==1:
            root.geometry("420x475")
            self.button_fuzz.grid()
            self.label_fuzz.grid()
            self.file_type_label.grid()
            self.rd_file_PL.grid(row=6, column=2, sticky=E, padx=4)
            self.rd_file_BS.grid(row=6, column=1, sticky=E, padx=4)
            self.frm_file_type.grid()

        else:
            root.geometry("455x560")
            self.rd1_no.grid(row=8, column=2, sticky=E, padx=4)
            self.rd_yes.grid(row=8, column=1, sticky=E, padx=4)
            self.frm_model.grid(column=1, row=7, sticky=E, pady=(0, 10))
            self.frm_radio.grid(column=1, row=8, sticky=E, pady=(0, 10))
            self.button_retrain.grid(column = 1, row =10, sticky = E, pady=(0, 15),padx=4)
            self.button1.grid(column=1,row=9,sticky=E,pady=(0, 15),padx=4)
            self.file1_label_retrain.grid(row=8, sticky=W,padx=4,pady=(0, 15))
            self.file1_select_model.grid(row=7, sticky=W,padx=4,pady=(0, 15))
            self.training_file_label.grid(row=9, sticky=W, padx=4, pady=(0, 10))
            self.retrain_label.grid(row=10, sticky=W, padx=4, pady=(0, 10))


    def retrain(self):
        try:
            if self.var.get() ==1:
                self.df_mapping = pd.read_excel(self.filename2.name, encoding='utf-8')
                self.df_mapp = self.df_mapping.iloc[:, 1:]
                print(self.df_mapp)
                cols = ['Keywords', 'Level 1', 'Level 2', 'Level 3', 'Level 4']
                self.df_mapp_new = self.df_mapp[cols].replace(',', ' ', regex=True)
                self.df_mapp_new['Combine_col'] = self.df_mapp_new[['Level 1', 'Level 2', 'Level 3', 'Level 4']].apply(lambda x: ','.join(x), axis=1)
                cols1=['Level 1', 'Level 2', 'Level 3', 'Level 4']
                self.df_mapp_new1 = self.df_mapp_new.drop(cols1, axis=1)
                print(self.df_mapp_new1)
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import CountVectorizer
                from sklearn.feature_extraction.text import TfidfTransformer
                #from sklearn.neighbors import KNeighborsClassifier
                from sklearn.svm import LinearSVC
                X_train, X_test, y_train, y_test = train_test_split(self.df_mapp_new1['Keywords'], self.df_mapp_new1['Combine_col'],random_state=4,test_size =0.1)
                count_vect = CountVectorizer()
                X_train_counts = count_vect.fit_transform(X_train)

                tfidf_transformer = TfidfTransformer()
                X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
                #knn = KNeighborsClassifier(n_neighbors=7)
                self.clf_svc_bs = LinearSVC().fit(X_train_tfidf, y_train)
                #self.clf_knn = knn.fit(X_train_tfidf, y_train)
                c = datetime.datetime.now()
                x = c.strftime('%d.%m.%Y')
                shutil.copy('BS.pkl', r'BS_%s.pkl' % x)
                os.remove('BS.pkl')
                print('Old model copied with datetime and deleted')
                with open('BS.pkl', 'wb') as f:
                    pickle.dump(self.clf_svc_bs, f)

                shutil.copy('count_vect_BS', r'count_vect_BS_%s' % x)
                os.remove('count_vect_BS')
                pickle.dump(count_vect, open('count_vect_BS', 'wb'))
                messagebox.showinfo('Model Trained','The new model has been created and trained')

            else:
                self.df_mapping = pd.read_excel(self.filename2.name, encoding='utf-8')
                self.df_mapp = self.df_mapping.iloc[:, 1:]
                cols = ['Keywords', 'Level 1', 'Level 2', 'Level 3', 'Level 4']
                self.df_mapp_new = self.df_mapp[cols].replace(',', ' ', regex=True)
                self.df_mapp_new['Combine_col'] = self.df_mapp_new[['Level 1', 'Level 2', 'Level 3', 'Level 4']].apply(lambda x: ','.join(x), axis=1)
                cols2 = ['Level 1', 'Level 2', 'Level 3', 'Level 4']
                self.df_mapp_new1 = self.df_mapp_new.drop(cols2, axis=1)
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import CountVectorizer
                from sklearn.feature_extraction.text import TfidfTransformer
                #from sklearn.neighbors import KNeighborsClassifier
                from sklearn.svm import LinearSVC
                X_train, X_test, y_train, y_test = train_test_split(self.df_mapp_new1['Keywords'],
                                                                    self.df_mapp_new1['Combine_col'], random_state=4)
                count_vect = CountVectorizer()
                X_train_counts = count_vect.fit_transform(X_train)
                tfidf_transformer = TfidfTransformer()
                X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
                self.clf_svc_pl = LinearSVC().fit(X_train_tfidf, y_train)
                #knn = KNeighborsClassifier(n_neighbors=7)
                #self.clf_knn = knn.fit(X_train_tfidf, y_train)

                c1 = datetime.datetime.now()
                x1 = c1.strftime('%d.%m.%Y')
                shutil.copy('PL.pkl', r'PL_%s.pkl' % x1)
                os.remove('PL.pkl')
                with open('PL.pkl', 'wb') as f:
                    pickle.dump(self.clf_svc_pl, f)

                shutil.copy('count_vect_PL', r'count_vect_PL_%s' % x1)
                os.remove('count_vect_PL')
                pickle.dump(count_vect, open('count_vect_PL', 'wb'))
                messagebox.showinfo('Model Trained', 'New model file has been created')
        except:
            messagebox.showerror('Retraining Model','Not able to train new model due to some error')

    def open_file1(self):
        try:
            self.filename1=filedialog.askopenfile(initialdir="/", title='Select File', filetypes=(('excel files', ".xlsx"), ("all files", "*.*")))
            #my_label1 = Label(root,text = self.filename1).grid()
            p = 0
            file1 = (self.filename1).name
            self.df1 = pd.read_excel(file1,encoding= 'utf-8')
            #print(self.df1)
            self.df1=self.df1.dropna()
            self.df1.reset_index(drop=True, inplace=True)
            print(self.df1)
            print(self.df1.columns.values)
            for columns in self.df1.columns.values:
                # self.selected_cols = self.lb.curselection()
                self.lb.insert(p, columns)
                p += 1
            self.lb.update_idletasks()
            messagebox.showinfo('Import','Datafile imported successfully')
        except:
            messagebox.showerror('Import','Please import your datafile')

    def open_file2(self):
        try:
            self.filename2 =filedialog.askopenfile(initialdir="/", title='Select File', filetypes=(('excel files', ".xlsx"), ("all files", "*.*")))
            file2 = (self.filename2).name
            print(file2)
            messagebox.showinfo('Imported Successfully','Training File imported Successfully')
        except:
            messagebox.showerror('Import Error','Please import Mapping File')

    def open_mapping_file(self):
        try:
            self.filename2 =filedialog.askopenfile(initialdir="/", title='Select File', filetypes=(('excel files', ".xlsx"), ("all files", "*.*")))
            file2 = (self.filename2).name
            print(file2)
            messagebox.showinfo('Imported Successfully','File imported Successfully')
        except:
            messagebox.showerror('Import Error','Please import Mapping file')

    def predict(self):
        try:
            if self.v1.get()==1:
                self.final_result_Fuzzy()
            else:
                if self.var.get()==1:
                    with open('BS.pkl', 'rb') as f:
                        SVM_LEAP_clf = pickle.load(f)
                        f.close()
                    count_vect = pickle.load(open('count_vect_BS', 'rb'))
                    #SVM_LEAP_clf = pickle.load(self.resource_path('SVM_LEAP.pkl'))
                    #count_vect = pickle.load(self.resource_path('count_vect_LEAP'))
                    self.result = []
                    select_columns_entry1 = self.file1_column.get()
                    self.df_data_get=self.df1[select_columns_entry1]
                    self.df_data_reset=self.df_data_get.reset_index(drop=True)

                    #for i in self.df1[select_columns_entry1][1:]:
                    for i in self.df_data_reset:
                        predicted = SVM_LEAP_clf.predict(count_vect.transform([i]))
                        self.result.append(predicted)

                    print(self.result)
                    self.df_result = pd.DataFrame(self.result)
                    self.df_result.columns = ['Model Predictions']
                    #self.df_final = pd.concat([self.df1[select_columns_entry1], self.df_result], axis=1)
                    self.df_final = pd.concat([self.df_data_reset, self.df_result], axis=1)
                    self.new = self.df_final["Model Predictions"].str.split(",", n=4, expand=True)
                    self.df_final["Level 1"] = self.new[0]
                    self.df_final["Level 2"] = self.new[1]
                    self.df_final["Level 3"] = self.new[2]
                    self.df_final["Level 4"] = self.new[3]

                    self.df_final_new = self.df_final.drop('Model Predictions', axis=1)
                    print(self.df_final_new)

                    self.final_result_ML()

                else:
                    with open('PL.pkl', 'rb') as f:
                        SVM_LEAP_clf = pickle.load(f)
                        f.close()
                    count_vect = pickle.load(open('count_vect_PL', 'rb'))
                    # SVM_LEAP_clf = pickle.load(self.resource_path('SVM_LEAP.pkl'))
                    # count_vect = pickle.load(self.resource_path('count_vect_LEAP'))
                    self.result = []
                    select_columns_entry1 = self.file1_column.get()
                    for i in self.df1[select_columns_entry1]:
                        predicted = SVM_LEAP_clf.predict(count_vect.transform([i]))
                        self.result.append(predicted)

                    print(self.result)
                    self.df_result = pd.DataFrame(self.result)
                    self.df_result.columns = ['Model Predictions']
                    self.df_final = pd.concat([self.df1[select_columns_entry1], self.df_result], axis=1)
                    self.new = self.df_final["Model Predictions"].str.split(",", n=4, expand=True)
                    self.df_final["Level 1"] = self.new[0]
                    self.df_final["Level 2"] = self.new[1]
                    self.df_final["Level 3"] = self.new[2]
                    self.df_final["Level 4"] = self.new[3]

                    self.df_final_new = self.df_final.drop('Model Predictions', axis=1)
                    print(self.df_final_new)

                    self.final_result_ML()
        except:
            messagebox.showerror('Predict Error','Some error occured while trying to predict the mappings\n Make sure the output file is not already open')

    def final_result_ML(self):
        try:
            self.selected_cols_list = []
            df_list_keywords_level4 = []

            self.selected_cols = self.lb.curselection()
            # print(self.selected_cols)
            for a in self.selected_cols:
                self.selected_cols_list.append(self.lb.get(a))

            for each_cols in self.selected_cols_list:
                self.df_selected = self.df1[each_cols]
                # print(self.df_selected)
                df_list_keywords_level4.append(self.df_selected)
            # print(df_list_keywords_level4)
            self.df_merge = pd.concat(df_list_keywords_level4, axis=1)
            self.merged_df1 = pd.concat([self.df_final_new,self.df_merge], axis=1)

            if self.var.get()==1:
                #export_csv = self.merged_df1.to_excel(os.getcwd()+r'output_ML.xlsx', index = None)
                export_csv = self.merged_df1.to_excel(r'Output_ML_BS.xlsx', index=None,sheet_name='ML_Output')
                messagebox.showinfo('Output Extracted','Please check output file in current folder')
            else:
                export_csv1 = self.merged_df1.to_excel(r'Output_ML_PL.xlsx', index=None, sheet_name='ML_Output')
                messagebox.showinfo('Output Extracted','Please check output file in current folder')
        except:
            messagebox.showerror('Error','Unable to generate final output due to some error')


    def fuzzy_match_keywords_fratio(self,k):

        global counter1
        print('Iterating on cell number: ',counter1)
        counter1+=1
        self.k=k
        print('Process is running ....please wait button to unfreeze')
        self.result1.append(process.extract(k, self.df_repo_string2, limit=1, scorer=fuzz.token_set_ratio))

    def multi_thread1(self):
        try:
            select_columns_entry1=self.file1_column.get()
            for i in self.df1[select_columns_entry1]:
                print(i)
            #self.filename2 = os.getcwd() + r'\tblMapping.xlsx'
            file2 = (self.filename2).name
            self.df2 = pd.read_excel(file2,encoding= 'utf-8')
            for k in self.df2['Keywords']:
                pass
                #print(k)
            self.cols1 = ['Keywords', 'Level 1', 'Level 2', 'Level 3', 'Level 4']
            self.new_repo2 = self.df2[self.cols1].replace(',', ' ', regex=True)
            self.df_repo_string2 = self.new_repo2.apply(",".join, axis=1)
            #print(self.df_repo_string2)
            self.result1 = []
            self.list_of_threads = list()
            select_columns_entry1 = self.file1_column.get()
            length = len(self.df1[select_columns_entry1])  #len(select_columns_entry1)
            k = iter(list(self.df1[select_columns_entry1].astype(str)))
            for j in range(len(self.df1[select_columns_entry1])):
                print('Appox. time 5-12 mins for 350 cells to compute')

                self.t = threading.Thread(target=self.fuzzy_match_keywords_fratio, args=(next(k),))
                self.list_of_threads.append(self.t)

                self.t.start()

                for self.t in self.list_of_threads:
                    self.t.join()

            self.selected_cols_list=[]
            df_list_keywords_level4=[]
            self.selected_cols = self.lb.curselection()
            #print(self.selected_cols)
            for a in self.selected_cols:
                self.selected_cols_list.append(self.lb.get(a))
                #self.col_names= self.lb.get(a)
            #print(self.selected_cols_list)

            for each_cols in self.selected_cols_list:
                self.df_selected=self.df1[each_cols]
                #print(self.df_selected)
                df_list_keywords_level4.append(self.df_selected)

            self.df_merge = pd.concat(df_list_keywords_level4,axis=1)
            #print(self.df_merge)

            self.final_result1 = []

            for values in self.result1:
                self.final_result1.append((',').join((values[0][0], str(values[0][1]), str(values[0][2]))))
            #print(self.final_result1)

            select_columns_entry1 = self.file1_column.get()
            self.matched_df1 = pd.DataFrame(self.final_result1)
            self.matched_df1.columns = ['Match_2']
            #df3 = pd.DataFrame(self.matched_df1)
            #self.merged_df1 = pd.concat([self.df1[select_columns_entry1], self.matched_df1,self.df_merge], axis=1)
            self.merged_df1 = pd.concat([self.df1[select_columns_entry1], self.matched_df1], axis=1)
            #print(self.merged_df1)
            self.new_df1 = self.merged_df1["Match_2"].str.split(",", n=6, expand=True)

            self.merged_df1["Keywords"] = self.new_df1[0]
            self.merged_df1["Level 1"] = self.new_df1[1]
            self.merged_df1["Level 2"] = self.new_df1[2]
            self.merged_df1["Level 3"] = self.new_df1[3]
            self.merged_df1["Level 4"] = self.new_df1[4]
            self.merged_df1["Score"] = (self.new_df1[5])
            self.merged_df1["Cell"] = self.new_df1[6]
            self.merged_df1.drop(columns=['Match_2'], axis=1, inplace=True)
            #print(self.merged_df1)

            self.result_match1 = []
            for score in self.merged_df1['Score']:
                if (int(score) >= 70):
                    self.result_match1.append('Match')
                else:
                    self.result_match1.append('Partial')

            self.result_match_df1 = pd.DataFrame(self.result_match1)
            self.result_match_df1.columns = ['Result']

            # def consolidated_result(self):
            self.merged_df1['Result'] = self.result_match_df1
            #print(self.merged_df1)
            cols_drop = ['Keywords','Cell','Result']

            self.final_df=self.merged_df1.drop(cols_drop,axis=1)
            self.final_l4=pd.concat([self.final_df,self.df_merge],axis=1)

        except:
            messagebox.showerror('Error Ocurred','Some error occured running Fuzzy match method')

    def fuzzy_match_all_cols(self,i):
        global counter
        print('Iterating on cell number: ',counter)
        counter+=1
        self.i =i
        print('Process is running ....please wait button to unfreeze')
        self.result.append(process.extract(i, self.df2_to_string, limit=1, scorer=fuzz.partial_ratio))

    def multi_thread(self):
        try:
            select_columns_entry1=self.file1_column.get()
            for i in self.df1[select_columns_entry1]:
                print(i)
        except:
            messagebox.showerror('Correct Column Name','Please enter correct column from Datafile')
        #self.filename2 = os.getcwd() + r'\tblMapping.xlsx'
        file2 = (self.filename2).name
        #self.df2 = pd.read_excel(self.filename2)
        self.df2 = pd.read_excel(file2)
        #print(self.df2)
        for k in self.df2['Keywords']:
            pass
            #print(k)
        self.cols = ['Keywords', 'Level 1', 'Level 2', 'Level 3', 'Level 4']
        self.new_repo = self.df2[self.cols].replace(',', ' ', regex=True)
        #print(self.df2)
        replace_df = self.df2[self.cols].replace(',', ' ', regex=True)
        self.df2_to_string = replace_df.apply(",".join, axis=1)
        #print(self.df2_to_string)
        #self.button.config(state= DISABLED)
        self.result = []
        self.list_of_threads = list()
        select_columns_entry1 = self.file1_column.get()
        length = len(self.df1[select_columns_entry1])  #len(select_columns_entry1)
        #print(length)
        #count = 0
        #batch_size = 15
        #while (length - (batch_size * count)) > 0:
        i = iter(list(self.df1[select_columns_entry1].astype(str)))
        for j in range(len(self.df1[select_columns_entry1])):
            print('Appox. time is 5-10 mins for 350 Cells')

            self.t = threading.Thread(target=self.fuzzy_match_all_cols, args=(next(i),))
            self.list_of_threads.append(self.t)

            self.t.start()
            for self.t in self.list_of_threads:
                self.t.join()
        #count += 1
                # block the execution of the main thread untill the join thread terminates

   # def get_consolidated_result(self):
        self.selected_cols_list_all_cols = []
        self.df_list_all_cols=[]

        self.selected_cols1 = self.lb.curselection()
        #print(self.selected_cols1)
        for a in self.selected_cols1:
            self.selected_cols_list_all_cols.append(self.lb.get(a))
            # self.col_names= self.lb.get(a)
        #print(self.selected_cols_list_all_cols)

        for each_cols in self.selected_cols_list_all_cols:
            self.df_selected1 = self.df1[each_cols]
            #print(self.df_selected1)
            self.df_list_all_cols.append(self.df_selected1)

        self.df_merge1=pd.concat(self.df_list_all_cols,axis=1)
        #print(self.df_merge1)
        self.final_result = []
        for values in self.result:
            self.final_result.append((',').join((values[0][0], str(values[0][1]), str(values[0][2]))))

        select_columns_entry1 = self.file1_column.get()
        self.matched_df = pd.DataFrame(self.final_result)
        self.matched_df.columns = ['Match_1']
        #self.merged_df = pd.concat([self.df1[select_columns_entry1], self.matched_df,self.df_merge1], axis=1)
        self.merged_df = pd.concat([self.df1[select_columns_entry1], self.matched_df], axis=1)
        #print(self.merged_df)
        self.new_df = self.merged_df['Match_1'].str.split(",", n=6, expand=True)
        #print(self.new_df)
        self.merged_df["Keywords"] = self.new_df[0]
        self.merged_df["Level 1"] = self.new_df[1]
        self.merged_df["Level 2"] = self.new_df[2]
        self.merged_df["Level 3"] = self.new_df[3]
        self.merged_df["Level 4"] = self.new_df[4]
        self.merged_df["Score"] = (self.new_df[5])
        self.merged_df["Cell"] = self.new_df[6]

        self.merged_df.drop(columns=['Match_1'], axis = 1, inplace = True)

        self.result_match = []
        for score_p in self.merged_df['Score']:
            if (int(score_p)>=70):
                self.result_match.append('Match')
            else:
                self.result_match.append('Partial')

        self.result_match_df = pd.DataFrame(self.result_match)
        self.result_match_df.columns = ['Result']
        #print(self.result_match_df)
        self.merged_df['Result'] = self.result_match_df
        #print(self.merged_df)
        cols_drop1=['Keywords','Cell','Result']
        self.final_df1 = self.merged_df.drop(cols_drop1, axis=1)
        self.final_all=pd.concat([self.final_df1,self.df_merge1],axis=1)

    def final_result_Fuzzy(self):

        try:
            self.multi_thread()
            self.multi_thread1()
            #df_OLD = self.final_df1
            df_OLD = self.final_all
            # df_NEW = pd.read_excel(self.f1.name).fillna(0)
            #df_NEW = self.final_df
            df_NEW = self.final_l4
            select_columns_entry1 = self.file1_column.get()
            select_columns_entry2 = self.file2_column.get()

            self.df_data_keywords = df_NEW[[select_columns_entry1, 'Level 4', 'Score']]
            #print(self.df_data_keywords)
            self.df_all_cols = df_OLD[[select_columns_entry1, 'Level 4', 'Score']]
            #print(self.df_all_cols)

            self.dfDiff = self.df_all_cols.copy()

            for row in range(self.dfDiff.shape[0]):
                for col in range(self.dfDiff.shape[1]):
                    value_OLD = self.df_all_cols.iloc[row, col]
                    try:
                        value_NEW = self.df_data_keywords.iloc[row, col]
                        if value_OLD == value_NEW:
                            self.dfDiff.iloc[row, col] = self.df_data_keywords.iloc[row, col]
                        else:
                            self.dfDiff.iloc[row, col] = ('{}-->{}').format(value_OLD, value_NEW)
                    except:
                        self.dfDiff.iloc[row, col] = ('{}-->{}').format(value_OLD, 'NaN')

            # self.fname = '{} vs {}.xlsx'.format(self.path_OLD, self.path_NEW)
            self.fname = 'Variance.xlsx'
            #print(self.fname)

            # print(path_NEW.stem)
            # print(path_OLD.stem)
            # fname = 'Variance.xlsx'
            writer = pd.ExcelWriter(self.fname, engine='xlsxwriter')
            self.dfDiff.to_excel(writer, sheet_name='Variance', index=False)
            df_OLD.to_excel(writer, sheet_name='Fuzzy Result-1', index=False)
            df_NEW.to_excel(writer, sheet_name='Fuzzy Result-2', index=False)


            workbook = writer.book
            worksheet = writer.sheets['Variance']
            worksheet.hide_gridlines(2)

            # define formats
            grey_fmt = workbook.add_format({'font_color': '#010d05'})
            highlight_fmt = workbook.add_format({'font_color': '#FF0000', 'bold': True})

            ## highlight changed cells
            worksheet.conditional_format('A1:ZZ1000', {'type': 'text',
                                                       'criteria': 'containing',
                                                       'value': '-->',
                                                       'format': highlight_fmt})
            ## highlight unchanged cells
            worksheet.conditional_format('A1:ZZ1000', {'type': 'text',
                                                       'criteria': 'not containing',
                                                       'value': '-->',
                                                       'format': grey_fmt})
            # save
            writer.save()
            workbook.close()
            print('Done')
            #messagebox.showinfo('Process', 'Process Completed')


            self.df3 = pd.DataFrame()
            self.df3.empty

            for b in self.final_all.index.values:
                if int(self.final_all['Score'][b]) >= int(self.final_l4['Score'][b]):
                    self.df_temp = self.final_all.iloc[[b]]
                    self.df3 = self.df3.append(self.df_temp.iloc[0], ignore_index=True)
                    self.df3=self.df3.reindex(self.df_temp.columns, axis=1)
                    del self.df_temp
                elif int(self.final_l4['Score'][b])>int(self.final_all['Score'][b]):
                    self.df_temp = self.final_l4.iloc[[b]]
                    self.df3 = self.df3.append(self.df_temp.iloc[0], ignore_index=True)
                    self.df3 = self.df3.reindex(self.df_temp.columns, axis=1)
                    del self.df_temp
                else:
                    print('*************')

            print(self.df3)
            if self.f1.get()==1:
                export_final_excel = self.df3.to_excel(os.getcwd()+'\Output_Fuzzy_BS.xlsx', index=None, sheet_name='BS_Fuzzy')
                messagebox.showinfo('Process Completed','Completed,please check Final_Output and Variance sheets')
            else:
                export_final_excel = self.df3.to_excel(os.getcwd() + '\Output_Fuzzy_PL.xlsx', index=None,sheet_name='PL_Fuzzy')
                messagebox.showinfo('Process Completed', 'Completed,please check Final_Output and Variance sheets')
        except:
            messagebox.showerror('Correct Multiple Errors','1. Please enter correct column from Datafile \n2. Please clean datafile before running the tool \n3. '
                                                               'Please make sure above buttons/options executed before running the tool')


root = Tk()
a = leap(root)
root.mainloop()
