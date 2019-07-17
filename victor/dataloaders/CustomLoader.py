import torch

class Dataset(data.Dataset):
    def __init__(self, root, x_field, y_field, start=0,stop=-1, Cout = 3, Cin=41, H=256, W=256):

        self.Cin     = Cin
        self.Cout    = Cout
        self.x_files = []
        self.y_files = []
        prefix       = 'epic_1b_'

        x_path = "{}/{}".format(root,x_field)
        y_path = "{}/{}".format(root,y_field)

        files = sorted(glob(x_path+"/**.npz"))
        files = files[start:stop]
        self.x_files += files

        # get matching y's
        for i,file in enumerate(files):
            path,b    = os.path.split(file) # split off basename
            timestamp = b[b.find(prefix) : b.find("_Collocated_MERRA2.npz")]
            y_file = "{}/{}.npz".format(y_path,timestamp)
            self.y_files.append(y_file)

        self.N = len(self.x_files)
        self.x = torch.zeros([self.N,self.Cin,H,W],dtype=torch.float)
        self.y = torch.zeros([self.N,self.Cout,H,W],dtype=torch.float)

    def __len__(self):
        return self.N

    def __getitem__(self,index):
        im_x = readin_epic_meto(self.x_files[index])
        im_y = readin_epic_img(self.y_files[index])

        x = torch.from_numpy(im_x).float()
        y = torch.from_numpy(im_y).float()
        return x, y

class CustomLoader:
    def __init__(self, opt):
        pass
