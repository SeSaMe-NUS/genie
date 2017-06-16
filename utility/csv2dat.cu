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
