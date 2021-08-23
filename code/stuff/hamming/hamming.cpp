// C++ implementation of above approach
#include <bits/stdc++.h>
#include <bitset>

using namespace std;


#define TRUE 1815043611u
#define FALSE 29794788u

// Function to calculate hamming distance
int hammingDistance(unsigned n1, unsigned n2)
{
	unsigned x = n1 ^ n2;
	int setBits = 0;

	while (x > 0) {
		setBits += x & 1;
		x >>= 1;
	}

	return setBits;
}

// Driver code
int main()
{
    srand(time(NULL));

    int hd;
    unsigned n1, n2;
    int max_hd = 0;
    unsigned max_n1, max_n2;
    int i=0;


    while(i++ < 1000){
        n1 = rand();
        n2 = rand();
        hd = hammingDistance(n1, n2);

        if(hd > max_hd){
            max_hd = hd;
            max_n1 = n1;
            max_n2 = n2;
        }
    }
    cout << "hd " << max_hd << endl;
    cout << "n1 n2 " << max_n1 << " " << max_n2 << endl;


    std::bitset<62> t(64);
    std::bitset<62> f(FALSE);
    std::cout << "t: " << t << '\n';
    std::cout << "f: " << f << '\n';
    cout << "HD " << hammingDistance(TRUE, FALSE) << endl;

	return 0;
}
