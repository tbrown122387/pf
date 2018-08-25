#include <stdio.h>
#include "svol_comparison.h"

int main(int argc, char **argv)
{
	if(argc == 1){
		printf("please enter the path to the data file");
	}else{
		run_svol_comparison(argv[1]);
	    //run_svol_comparison("/home/taylor/ssm_examples/data/some_csvs/svol_y_data.csv");

	}
	return 0;
}
