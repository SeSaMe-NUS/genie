// csv2dat.cu: This file can be compiled to an executable csv2dat.
// Use csv2dat to turn csv files to binary files.
//                         Convert usage:  ./convert [-a|-t] csvfilename binaryfilename
//                                 -a: append writing
//                                 -t: trunc writing
//                                 default: trunc writing
//
// The binary file: Every binary file consists of 4 consecutive parts. 
//                 The first part: item_num, length = sizeof(unsigned int) ------ hold the number of data item
//                 The second part: row_num, length = sizeof(unsigned int) ------ hold the number of row of data
//                 The third part: data, length = sizeof(int) * item_num ------ hold actual data array
//                 The fourth part: index, length = sizeof(unsigned int) * row_num ----- hold row index array,
//                                  which help determine the beginning of each row

#include "GPUGenie.h"

using namespace GPUGenie;
using namespace std;

int main(int argc, char **argv){
    if(argc < 3 || argc > 4 || (argc==4&&argv[1][0]!='-') || (argc==4&&argv[1][1]!='a'&&argv[1][1]!='t')){
        printf("Arguments Error: should be : command csvfile binaryfile\n");
        printf("Options: \n");
        printf("\t-a:\tfor app-writing");
        printf("\t-t:\tfor trunc-writing");
        printf("default:\t trunc-writing");
        return 0;
    }
    printf("Start converting...\n");
    if(argc == 3){
        cout<<"trunc writing"<<endl;
        csv2binary(argv[1],argv[2]);
    }    

    if(argc==4){
        if(argv[1][1]=='t'){
            cout<<"trunc writing"<<endl;
            csv2binary(argv[2],argv[3],false);
        
        }else{
            cout<<"append writing"<<endl;
            csv2binary(argv[2],argv[3],true);
            
        }
    }
    printf("Done!\n");
    return 0;

}
