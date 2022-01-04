'''
    This file extracts the science data from the space-based plasma spectrometer
    
    It cleans and filters the data, then exports it for plotting.
    
    Created by Sachin A. Reddy

    Dec 2021.

'''

import glob 
import os 
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None) #or 10 or None

path = (r'/Users/sr2/OneDrive - University College London/PhD/Research/'
        'Missions/SOAR/In-flight/Dec-21/20211204/data')
        #'Missions/SOAR/In-flight/Nov-21/20211113')

pkt_header_size = 44 #Phoenix is 22, CIRCE is 90, SOAR is 44, ROAR is 22
pkt_format = '*.bin' #Phoneix is .dat, CIRCE is .pkt, SOAR is .bin

class extractScience():

    try:
        def open_packets(self, path):
            
            #Opens the packet into a 8b / 8B format
            open_8_bit = []
            for filename in glob.glob(os.path.join(path, pkt_format)):      
                with open(os.path.join(os.getcwd(), filename), 'rb') as f:
                    open_bits = list("{0:08b}".format(c) for c in f.read()) #Opens as hex
                    open_8_bit.append(open_bits)
            
            print(f'Num of {pkt_format} files: ', len(open_8_bit))

            flatten_pkts = [i for j in open_8_bit for i in j]
            num_of_pkts = (len(flatten_pkts)//(pkt_header_size+174))
            print(f'Num of pkts: ', num_of_pkts)

            indi_pkt = np.array_split(flatten_pkts, num_of_pkts)
            
            return indi_pkt
    except RuntimeError:
            raise Exception('Problems opening packets')
    
    try:
        def split_packets(self, pkts):

            #Splits the pkts into a header and 'main'
            def split_head_main(self, pkts, start, stop):
                split_pkt = []
                for x in pkts:
                    pkt_index = list(x[start:stop])
                    split_pkt.append(pkt_index)

                return split_pkt
            
            head = split_head_main(self, pkts, 0, pkt_header_size+1)
            main = split_head_main(self, pkts, pkt_header_size, None)
            
            #Removes non-science pkts
            def identify_sci_only(self, pkt_split, index, start, stop):
                sci_only = []
                for y in pkt_split:
                    if y[index] == '00001000': #08 in hex
                        sci_index = list(y[start:stop])
                        split_join = (list(j for j in "".join(j for 
                                j in sci_index)))
                        sci_only.append(split_join)
                return sci_only
                
            sci_only_head = identify_sci_only(self, head, pkt_header_size, 
                    0, pkt_header_size)
            sci_only_main = identify_sci_only(self, main, 0 , None, None)

            if not sci_only_main:
                print('There are no science pkts')
            else:
                print('Num of sci pkts:', len(sci_only_main))

            return sci_only_head, sci_only_main
    
    except RuntimeError:
            raise Exception('Problems splitting packets')
    

extraction = extractScience()
total_pkts = extraction.open_packets(path) #open files
sci_head, sci_main = extraction.split_packets(total_pkts)

class convertScience():

    #1st stage (conversion functions)
    #-----------
    try:
        def usigned32(self, binary_str):
            as_bytes = int(binary_str, 2).to_bytes(4, 'big') #4 = bytes / 32 = 8-bit
            return int.from_bytes(as_bytes, 'big', signed=False)

        def bin2int(self, ieee_32):

            def mantissa(mantissa_str):
                power_count = -1
                mantissa_int = 0
                    
                for i in mantissa_str:
                    mantissa_int += (int(i) * pow(2, power_count))
                    power_count -= 1
                return (mantissa_int + 1)

            #bin2int function
            sign_bit = int(ieee_32[0])
            exp_bias = int(ieee_32[1:9], 2)
            exp_unbias = exp_bias - 127
            mantissa_str = ieee_32[9:]
            mantissa_int = mantissa(mantissa_str)
            real_no = pow(-1, sign_bit) * mantissa_int * pow(2, exp_unbias)
            return real_no

        def simpleInt(self, x):
            return int(x,2)
        
        def prep_binary(self, bin_data, start, stop, converter):

            binary_grouped = []
            for i in bin_data:
                index = str("".join(j for j in i[(start + 0):(start + stop)]))
                group_binary = [index[k:k+stop] for k 
                        in range(0, len(index), stop)] 
                binary_grouped.append(group_binary)

            bin2readable = [i for j in ([converter(x) for x in i] 
                    for i in binary_grouped) for i in j]
            return bin2readable

        def counts2energy(self, volt, index, c_id, prep_binary):
            
            #multi = 0.93 #30 min script
            multi = 0.92 #commissioning script

            counts = {}
            for i in range(16):
                energy_ind = str(round((volt * multi ** i) * k_factor, 2))
                nv = index + i*12
                #data, start index, end index, conversion function
                counts[energy_ind] = prep_binary(sci_main, nv, 12, self.simpleInt)
                counts = pd.DataFrame(counts) 
            counts['c_id'] = c_id       
            return counts
    
    except RuntimeError:
            raise Exception('Problems with conversion functions')
    
    #2nd stage (transformation functions)
    #-----------
    try:
        def unix2utc(self, utc):
            from datetime import datetime
            timestamp = datetime.fromtimestamp(utc)
            return timestamp

        def eci2lla(self, x,y,z,dt):
            #https://groups.google.com/g/astropy-dev/c/AIdMCZykFtw?pli=1
            #from datetime import datetime
            from astropy.coordinates import GCRS, ITRS, EarthLocation
            from astropy.coordinates import CartesianRepresentation
            from astropy import units as u

            # Read the coors in the Geocentric Celestial Reference System
            gcrs = GCRS(CartesianRepresentation(x=x*u.m,
                    y=y*u.m,z=z*u.m), obstime=dt)

            # Convert it to an Earth-fixed frame
            itrs = gcrs.transform_to(ITRS(obstime=dt))
            el = EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)

            # conversion to geodetic
            lon, lat, alt = el.to_geodetic()

            #Some weird bug that outputs as datetime, unless you use a df
            df = {'lat':lat,'lon':lon,'alt':alt}
            df = pd.DataFrame(df, index=[0])

            lat = df['lat'].values.tolist()
            lon = df['lon'].values.tolist()
            alt = df['alt'].values.tolist()

            return lat, lon, alt

        def check_quarts(self, df):
            df ['q_mag'] = df['q1']**2+df['q2']**2+df['q3']**2+df['q4']**2
            return df #result needs to be 1
    
    except RuntimeError:
            raise Exception('Problems with transformation functions')

    #build functions
    try:
        def build_header(self):

            #Header data
            #data, start index, end index, conversion function
            #-----------
            unix = self.prep_binary(sci_head, 320, 32, self.usigned32)
            utc = list(map(self.unix2utc, unix))

            #Attitude quarternion wrt to ECI
            q1 = self.prep_binary(sci_head, 0, 32, self.bin2int)
            q2 = self.prep_binary(sci_head, 32, 32, self.bin2int)
            q3 = self.prep_binary(sci_head, 64, 32, self.bin2int)
            q4 = self.prep_binary(sci_head, 96, 32, self.bin2int)

            #Rotational speed around the 3-axis in the ECI
            w_x = self.prep_binary(sci_head, 128, 32, self.bin2int)
            w_y = self.prep_binary(sci_head, 160, 32, self.bin2int)
            w_z = self.prep_binary(sci_head, 192, 32, self.bin2int)

            #position in the ECI
            pos_x = self.prep_binary(sci_head, 224, 32, self.bin2int)
            pos_y = self.prep_binary(sci_head, 256, 32, self.bin2int)
            pos_z = self.prep_binary(sci_head, 288, 32, self.bin2int)

            #Create df
            df = {'utc':utc,'posx':pos_x,'posy':pos_y,'posz':pos_z,
                    'q1':q1, 'q2':q2,'q3':q3, 'q4':q4}
            #        'wx':w_x,'wy':w_y, 'wz':w_z}
            df = pd.DataFrame(df)

            #Tranform df
            df[['lat','lon','alt']] = df.apply(lambda x: 
                self.eci2lla(x.posx, x.posy, x.posz, x.utc),
                axis = 1, result_type='expand')
            df['lat'] = df['lat'].str[0]
            df['lon'] = df['lon'].str[0]
            df['alt'] = df['alt'].str[0]

            df['alt'] = (df['alt'] - 6731) / 1000
            #df = self.check_quarts(df) #needs to = 1
            df = df.drop(['posx','posy','posz','q1','q2','q3','q4'],axis=1)

            return df

    except RuntimeError:
            raise Exception('Problems building header')

if not sci_head:
    print('No science pkts')
else:

    convert = convertScience()
    head = convert.build_header()

    #Main data
    #voltage, index, channel id, conversion function
    k_factor = 1.7

    n0v, n1v, n2v = 4, 2.4, 2.75
    n0 = convert.counts2energy(n0v, 42, 'n0', convert.prep_binary)
    n1 = convert.counts2energy(n1v, 234, 'n1', convert.prep_binary)
    n2 = convert.counts2energy(n2v, 426, 'n2', convert.prep_binary)

    i0v, i1v, i2v = 1.80, 3.162, 3.613
    i0 = convert.counts2energy(i0v, 628, 'i0', convert.prep_binary)
    i1 = convert.counts2energy(i1v, 820, 'i1', convert.prep_binary)
    i2 = convert.counts2energy(i2v, 1012, 'i2', convert.prep_binary)

    #join header and main data
    nh0 = pd.concat([head,n0], axis=1)
    nh1 = pd.concat([head,n1], axis=1)
    nh2 = pd.concat([head,n2], axis=1)

    ih0 = pd.concat([head,i0], axis=1)
    ih1 = pd.concat([head,i1], axis=1)
    ih2 = pd.concat([head,i2], axis=1)

    def sort_df(df):
        df = df.sort_values(by='utc', ascending=True)
        df = df.reset_index().drop(columns='index')
        return df

    nh0 = sort_df(nh0)
    nh1 = sort_df(nh1)
    nh2 = sort_df(nh2)
    ih0 = sort_df(ih0)

    #neutral = pd.concat([nh0, nh1, nh2], axis=1)
    
    print(nh0)

    #print(neutral)
    #print(ih0)
    #print(nh2)
    
    #print(ih0)
    #print(nh2)

    #ih0 = sort_df(ih0)
    #print(ih0)
    #print(ih2)

    class cleanScience():

        def melt_df(self, df, date):

            if type(date) == str:
                df = df[df['utc'] == date] 
            elif date == None:
                df = df
            else:
                df = df.iloc[date-1:date,:]
            
            #df = df.iloc[10:]
            #print(df)
            
            ebins = df.iloc[:,4:-1:] 
            cols = ebins.columns
            
            df = df.reset_index().melt(id_vars=['utc','c_id','lat','lon','alt'], value_vars=cols)
            df = df.rename (columns ={'variable':'energy','value':'counts'})
            df = df.astype({"counts":int, 'energy':float})

            #df = df[df['counts'] != 0] 
            return df

    #date = "2021-12-03 17:28:04"
    date = None
    clean = cleanScience()
    n0_melt = clean.melt_df(nh0,date)
    n1_melt = clean.melt_df(nh1,date)
    n2_melt = clean.melt_df(nh2,date)

    i0_melt = clean.melt_df(ih0,date)
    i1_melt = clean.melt_df(ih1,date)
    i2_melt = clean.melt_df(ih2,date)

    #ion_neutral = pd.concat([n0_melt], axis=0)
    #print(ion_neutral)

    ion_neutral = pd.concat([n0_melt,n1_melt,n2_melt,i0_melt], axis=0)
    ion_neutral = ion_neutral.sort_values(['utc','c_id','energy'], ascending =[True, True, True]) #Sort'''
    ion_neutral = ion_neutral.reset_index().drop(columns=['index'])
    #print('ion neutral\n',ion_neutral)

    output_pathfile = path + "/" + os.path.basename(path) + ".csv" # -4 removes '.pkts' or '.dat'
    ion_neutral.to_csv(output_pathfile, index = False, header = True)
    #print('exported')

    # import seaborn as sns
    # from matplotlib import pyplot as plt
    # utc = ion_neutral['utc'].iloc[0]
    # plt.figure(figsize=(5.5,3.5), dpi = 90)
    # hue, style = 'c_id', None
    # plt.title(f'{utc} // n0: {n0v}V, i0: {n2v}V', fontsize=10.5)
    # ax = sns.lineplot(data=ion_neutral, x = 'energy',y='counts', 
    #         hue = hue, style=style, legend=True)
    # ax.set_ylabel('Counts [per 10 ms]')
    # #ax.set_xlabel('Voltage [V]')
    # ax.set_xlabel('Energy [eV]')
    # ax.set_yscale('log', base=10)
    # plt.tight_layout()
    # plt.show()


    
