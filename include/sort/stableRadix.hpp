//abstractly
//universe scalar based on the moroton key resolution
//radix to sort morton code
	//create bit mask
	//prefix scan
		//up sweep
		//reset a[n-1] to 0
		//down sweep
	//map from i -> j
//swap buffers

//Z-order space filling curve
	//scale positions into the 2^bit_count resolution
		//resolution = 10  (0x03FF is the 10 bits)
		//scale = 2^resolution
		//x_scale = posx[i] - world_min
		//x_scale /= world_max
		//xi = static_cast<uint32_t>(xi * scale)
	//collate particle xyz positions by taking the top 10 bits per dim
		//buffer[i] = posz[i] << 20 | posy[i] << 10 | posx[i] -> guarenteed to only be 10 bits of value
	//order morton keys in batches of n where 3*resolution % n = 0 IE 3 or 6 using radix

	//radix a mapping from i -> j
		//create bit mask of predicate bit[k] == 1 ? 0 : 1 IE = !((bit[k] >> k) & 0x01)
		//prefix scan
			//we copy the is_zero into global memory
			//upsweep
				//image each A member as a leaf of a tree
				//we then add 2 leafs to create a combined node value above
				//then we add those two nodes and work out way to root where we have some what fragmented knowledge
				//say the n/2th index has how many zeros in the lhs adn the n-2th has the number in the rhs.
				//the root being the addition of these two forms the total number
				//so now we need to organise this so get a linear count
			//reset root value to 0, this does a semantic flip and youll see why this is needed before the down sweep
				//since the a[7] was our final root storing how many total zeros, it will always be the rhs adding the previous fragment earlier to it
				//so it will end up having all the previous fragments added to it
			//down sweep
				//so we have fragmented data where lhs has the number of zeros from 0
				//a lhs node has the zeros from whatever the index of its smalelst leaf is
				//so at the 1th layer of the tree, that lhs has its most lhs node is [0]
				//where as if u go down another layer, that lhs smallest leaf was a[2]

				//so each rhs needs to have whats on the lhs of it added. and since rhs will inherit the parent, and add the old lhs we now have
				//both from its smallest leaf say a[2] and the previous parts of the array say fomr a[0]
			
		//scatter where we map from i -> j
			//ideally we want to create two buckets
			//a bucket with all the 0s and a bucket with all the 1s
			//so we map our 0s to where they are out of all the zeros. 
			//ask am i the 2th 0 or the 4th 0 instance ya know
			//we do the same for the 1s
			//then after we have all the 1s and all the 0s in order of there sorta occurence
			//we push the 1s up the width of the 0s bucket so the 1s begin right after the 0s finish all inside of a common arr



