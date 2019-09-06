# Similar City Finder
<img src="https://github.com/hsuetsugu/similar-city-finder/blob/master/img/city_roadnetwork.png" alt="top">

Road-network image as the representation of city feature

# Question
What makes us think that city A looks like city B?

# Hypothesis
Road-networks play an important role in making an impression on cities

# Methodology
* Input data
    * 100,000 road-network images about 13,000 cities
    * Network statics
        * number of streets
        * number of intersections
        * number of streets per intersection 
* Image-based approach
    * Dimension reduction
        * Convolutional Auto-Encoders (CAE)
    * Distance calculation
        * Euclidean distance
* Network statistics-based approach
    * Distance calculation
        * JS divergence for number of streets per intersection
        * Euclidean distance
* Distance calculation
    *  Euclidean distance


# Result examples : New York(as an example of Orthogonal shape)
<img src="https://github.com/hsuetsugu/similar-city-finder/blob/master/img/NewYork.png" alt="top">


# Result examples : Tokyo(as an example of Circular shape)
<img src="https://github.com/hsuetsugu/similar-city-finder/blob/master/img/Tokyo.png" alt="top">


# Thanks to :
    * osmnx : great library for obtaining geo-related information

