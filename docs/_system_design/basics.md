---
layout: page
title: System Design Basics
---


# Scalability

What is it that we really mean by scalability? 

A service is said to be scalable if when we increase the resources in a system, it results in increased performance in a manner proportional to resources added. 
Increasing performance in general means serving more units of work, but it can also be to handle larger units of work, such as when datasets grow.

In distributed systems there are other reasons for adding resources to a system; for example to improve the reliability of the offered service. Introducing redundancy is an important first line of defense against failures. An always-on service is said to be scalable if adding resources to facilitate redundancy does not result in a loss of performance.

Source [All Things Distributed](https://www.allthingsdistributed.com/2006/03/a_word_on_scalability.html)

### Horizontal vs Vertical Scaling

Horizontal scaling means that you are adding more servers -- cloning server.

Vertical scaling means that you are adding more power to your existing server -- adding CPU, RAM, storage etc. 

![Alt text](https://www.cloudzero.com/hubfs/blog/horizontal-vs-vertical-scaling.webp)


#### Advantages of horizontal scaling

* **Scaling is easier from a hardware perspective** - All horizontal scaling requires you to do is add additional machines to your current pool. It eliminates the need to analyze which system specifications you need to upgrade.
* **Fewer periods of downtime** - Because you’re adding a machine, you don’t have to switch the old machine off while scaling. If done effectively, there may never be a need for downtime and clients are less likely to be impacted.
* **Increased resilience and fault tolerance** - Relying on a single node for all your data and operations puts you at a high risk of losing it all when it fails. Distributing it among several nodes saves you from losing it all. 
* **Increased performance** - If you are using horizontal scaling to manage your network traffic, it allows for more endpoints for connections, considering that the load will be delegated among multiple machines.     

#### Disadvantages of horizontal scaling

* **Increased complexity of maintenance and operation** - Multiple servers are harder to maintain than a single server is. Additionally, you will need to add software for load balancing and possibly virtualization. Backing up your machines may also become a little more complex. You will need to ensure that nodes synchronize and communicate effectively. 
* **Increased Initial costs** - Adding new servers is far more expensive than upgrading old ones.   

#### Advantages of vertical scaling

* **Cost-effective** - Upgrading a pre-existing server costs less than purchasing a new one. Additionally, you are less likely to add new backup and virtualization software when scaling vertically. Maintenance costs may potentially remain the same too.
* **Less complex process communication** - When a single node handles all the layers of your services, it will not have to synchronize and communicate with other machines to work. This may result in faster responses.
* **Less complicated maintenance** - Not only is maintenance cheaper but it is less complex because of the number of nodes you will need to manage. 
* **Less need for software changes** - You are less likely to change how the software on a server works or how it is implemented.         

#### Disadvantages of vertical scaling

* **Higher possibility for downtime** - Unless you have a backup server that can handle operations and requests, you will need some considerable downtime to upgrade your machine. 
* **Single point of failure** - Having all your operations on a single server increases the risk of losing all your data if a hardware or software failure was to occur. 
* **Upgrade limitations** - There is a limitation to how much you can upgrade a machine. Every machine has its threshold for RAM, storage, and processing power. 

Source [CLOUDZERO](https://www.cloudzero.com/blog/horizontal-vs-vertical-scaling)


# Reliability
It is the probability a system will fail in a given period.A system is said to be reliable, if it delivers its required service even after a component (software/hardware) fails. In any distributed system, any failing component can be replaced by another healthy component. Ensuring the completion of the requested task.

# Availability
It is the time a system remians operation to perform its required function in a specific period. It is a simple measure of the percentage of time that system, service or a machine remains operation under normal circumstances. An aircraft that can be flown for many hours a month without much
downtime can be said to have a high availability. 

#### Reliability vs Availability
If a system is reliable, it is available. A system can reach high availability without being reliable by minimizing repair time and ensuring that spares are always available when needed.

# Load Balancing
It distributes incoming requests to available servers inorder to improve reponsiveness and availability of applications. Load balancers are effective at:
* _Preventing requests from going to unhealthy servers._
* _Preventing overloading resources._
* _Helps to eliminate a single point of failure._

Load balancers can be placed at 3 places:
* _Between user and web server._
* _Between web server and internal platform layers like application servers._
* _Between internal platform server and database._

Additional benefits:
* **SSL termination** - Decrypt incoming requests and encrypt server respones so backend servers don not have to perform these potentially expensive operations.
* **Session persistence** - Issue cookies and route a specif client's requests to same instance if the web apps do not keep track of sessions.

_Load balancer can be a single point of failure. To overcome this, a second load balancer (redundant load balancer) is is connected to form a cluster. The second load balancer monitors the health of the primary Load balancer and if the primary fails, the secondary takes over._

#### Load Balancing Algorithms
Load balancers should only forward traffic to healthy server nodes. To makes sure whether a server is up, they perform regular health checks to make sure that the server is listening. If a servers fails a health check, it is removed from the pool and traffic is not sent towards it till the server is back up again.

* **Least Connection Method** - Directs traffic to servers with the fewest active connections. Useful for systems that require a large number of persistent client connections that are unevenly distibuted between servers.
* **Least Respone Time** - Directs traffics to servers that have the fewest active connections and the lowest response times.
* **Least Bandwidth Method** -  Directs traffic to server that is serving the least amount of traffic measured in megabits per second (Mbps)
* **Round Robin Method** - Cycles through the list of available servers, sending each new request to the next server.
* **Weighted Round Robin** - Cycles through the list of available servers while considering the processing the capabilities of the servers as well. Each server is assigned a weight based on the processing capacity.
* **IP HASH** - A hash of the IP address of the client is calculated to redirect the request to a server.

# Caching
Improves page load times and can reduce the load on your servers and databases. A cache is like short-term memory: it has limited amount of space, but is typically faster than the original data and contains the most recently accessed items. 

Databases often benefit from a uniform distribution of reads and writes across its partitions. Popular items can skew the distribution, causing bottlenecks. Putting a cache in front of a database can help absorb uneven loads and spikes in traffic.

#### Client Caching
Caches can be located on the client side(OS or browser), server side (reverse proxy), or in a distinct cache layer.

#### CDN Caching
Are mostly used for sites using large amounts of static media like images, videos, CSS, javascript files etc.

Considerations of using a CDN:
* **Cost**: CDNs are run by third party providers, and you are charged for data transfers in and out of the CDN. Caching infrequently used assets provides no significant benefits.
* **Setting an appropriate cache expiry**: For time sensitive content, setting a cached expiry time is important . If it is too long, the data might be old. if it is too short, it needs constant refreshing.
* **Invalidating files**: Remove files before they expire using - 
..* _Invalidate the CDN object by using the API provided by the vendor._
..* _Use object versioning to serve a different version of the object._

#### Cache updating strategies
##### Cache-aside 

The application is responsible for reading and writing storage. The cache does not interact with the storage directly.

![Alt text](https://github.com/donnemartin/system-design-primer/blob/master/images/ONjORqk.png?raw=true)
[Source](https://www.slideshare.net/tmatyashovsky/from-cache-to-in-memory-data-grid-introduction-to-hazelcast)

* Look for entry in cache, resulting in a cache miss
* Load entry from database
* Add entry to cache
* Return entry

```
def get_user(self, user_id):
    # look up query in cache
    user = cache.get("user.{0}", user_id)
    # if empty
    if user is None:
        # lookup database
        user = db.query("SELECT * FROM users WHERE user_id = {0}", user_id)
        if user is not None:
            key = "user.{0}".format(user_id)
            # update cache
            cache.set(key, json.dumps(user))
    return user
```

Memcached is generally used in this manner.

**Disadvantages:**
* Each cache miss results in 3 trips, which causes noticaeble delay.
* Data can become if it is updated in the database. This can be mitigated by adding a time-to-live(TTL) which forces an update of the cache.
* When the node fails, it is replaced by a new, empty node, increasing latency.


##### Write-through

The application uses the cache as the main data store, reading and writing to it. The cache is reponsible for reading and writing to the database.

![Alt text](https://github.com/donnemartin/system-design-primer/blob/master/images/0vBc0hN.png?raw=true)

[Source](https://www.slideshare.net/jboner/scalability-availability-stability-patterns/)

* Application add/updated entry in cache.
* Cache synchronously writes entry to data store.
* Return

Write-through is a slow operation due to the write operatio, but subsequent reads are fast. Users are generally more tolerant to latency while updating data than reading. Data is not stale.

**Disadvantages:**
* When a new node is created due to failure or scaling, the new node will not cache entries until the entry is updated in the database. Cache-aside in conjuction with write through can mitigate this issue.
* Most data written might never be read, which can be minimized by TTL.


##### Write-behind(write-back)

![Alt text](https://github.com/donnemartin/system-design-primer/blob/master/images/rgSrvjG.png?raw=true)

[Source](https://www.slideshare.net/jboner/scalability-availability-stability-patterns/)

* add/update entry in the cache
* asynchronously write entry to the database.

**Disadvantages:** 
* There could be data loss if the cache goes down prior to its contents hitting the data store.
* It is more complex to implement write-behind than cache-aside or write-through.


#### Cache eviction policies

1. **First in first out (FIFO):** The cache evicts the first block accessed without any regard to how often it was accessed before.
2. **Last in first out (LIFO):**  The cache evicts the block accessed most recently first without any regard to how many times it was accessed before.
3. **Least recently used (LRU):**  Discards the least recently used item first.
4. **Most recently used (MRU):** Discards the most recently used item first.
5. **Least Frequently used(LFU):** Discards the least frequent items first. Stores count/frequencies. 
6. **Random Replacement:** Randomly selects a candidate and discards it to free up space.
 